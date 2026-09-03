# 5. THE PROGRAMME — the development plan for Consequence-Equilibrium Attention

*MYCROFT (chair), 2026-09-03, HEAD `207e7b9`, branch `v17k-gate0`. This section is the body of the paper (`THESIS_CORRECTIONS_2.md` §0: the deliverable is the plan) and is written to be read standalone, repeatedly, over months, by one developer working alone in evenings on the certified RTX 4060 Laptop (`7.996 GiB`, torch `2.5.1+cu121`, `READ COSTS.md:53-63`). It merges the six roadmaps — `plan/plan_jupiter.md` (Lean, `J-`), `plan_saturn.md` (instrument, `S-`), `plan_mercury.md` (price, `M-`), `plan_mars.md` (attack, `R-`), `plan_venus.md` (prediction, `V-`), `plan_neptune.md` (systems, `N-`) — deduplicated by content, with every planet's card kept and its aliases printed on the card's header line. The judged sections (`judge/sec_shape.md`, `judge/sec_obstructions.md`, `judge/proposition_ledger.md`, `judge/sec_apparatus.md`, `judge/bind_ledger.md`) outrank the roadmaps and the designs wherever they differ; where a roadmap's number was overruled, the card carries the ruling.*

*Evidence classes on every load-bearing claim. `RUN` — executed on this box this session or in a named planet's session, carried with its geometry; `READ path:line` at HEAD `207e7b9`; `CITED [V]` (primary page fetched, title matched) or `[U]` (reached through an index) by canonical `references.bib` key; `DERIVED` with the steps shown; `[FITTED]` with its R-squared, `[ASSUMED]` with its reason (`READ COSTS.md:16-20`). No number appears without one of these. Struck constants of `STRUCK.md` appear nowhere. Nothing here is a result: no CEQ arm has been trained, BED-S has no cell, and every BED-S figure below is a floor formula or a design constant.*

*Two id spaces collide and are disambiguated once: `CONTRACT.md` D-1 is the dependency law (work is a DAG), `MISTAKES.md` D-1 is the design mechanism (racing a proven optimum). Mechanisms cited as bare `D-n / M-n / P-n / V-n` are `MISTAKES.md`'s 66 headings (`RUN grep -cE '^### (V|P|M|D)-[0-9]+'` reads `66`, including the lettered `V-14a` at `MISTAKES.md:770`); laws are cited by their `L-` name and rulings by number.*

---

## 5.0 The shape, and the claim the programme can at best earn

**The shape, in one sentence.** Consequence-Equilibrium Attention is one causal, row-stochastic attention operator read through a single triangular solve,
$$z=(I-\gamma W)^{-1}V,\qquad \Pi_\gamma=(1-\gamma)\,W(I-\gamma W)^{-1},\qquad O=\Pi_\gamma V,$$
over the record's Lean-checked three-corner base $W_{\beta,qk,m,\theta}$ (`READ lean/CEQ/V16Domain.lean:92-97`, `READ ceq/arm_smprime.py:144`), with a declared value-zero sink set $\mathcal{A}_{\rm sink}=\{0\}$, a goal set $\mathcal{A}_0$ and $K\ge 2$ constraint sets $\mathcal{A}_1,\dots,\mathcal{A}_K$ written as identity rows, so that one solve returns the jointly determined state $z$, its displacement $\Delta z$ under an intervention, and the reach-avoid committor $q^{(\bullet)}$ of every transient position; at $\gamma=0$ the read is softmax bitwise (`RUN[coord]` `torch.equal` `True` at $s=64$, $d=16$; rejection region $\max|O(0.5)-O(0)|=2.3002850040264393$).

**The claim the programme can at best earn, in one sentence.** On BED-S — a registered bed whose environment chain is a random DAG in token order, hence inside the arm's own operator class — CEQ reads the reach-avoid vector below the exact restricted-view floor and the displacement field above depth-1 softmax's field cosine at matched depth and parameters, with an exact certificate ($\delta=0$ on the solve route), a trained $\hat\gamma$ that is MOVED under the boundary null $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ (95 per cent point $2.7055$, `RUN[J]`), and a separation from ChaCAL-with-a-sink-token that a column device does not reproduce — and it can never earn "beats softmax" on any bed where a deeper softmax stack is a fellow approximator that computes the same resolvent by iteration (`wang-2024-incontext-td`, `xie-2026-softmax-rl` `[V]`; R-SKY, `READ CEQ_V16_CONTRACT.md:209`).

Everything the programme can lose is written down beside what it can win: §5.10's outcome trees name the sentence each ending licenses and the replacement route it owes.

---

## 5.1 How to use this plan

**The phases and their gates.** Seven phases. Each ends in a decision the author takes alone, from printed numbers, without an agent.

| phase | goal | the gate at its end | the numbers that decide it |
|---|---|---|---|
| **0** free work: theorems, manifests, specifications, censuses | no arm trains before its identity theorems are green (L-LEAN, `READ CEQ_V15_CONTRACT.md:57-58`) | `lake build` exit `0`, zero `sorry`, axioms `[propext, Classical.choice, Quot.sound]` only; `5/5` manifest drift flips move the hash; every obstruction sentence carries its vacuity inequality |
| **1** BED-S construction and admission | D-4 admission: a bed not admitted produces no reading | every label sd $>0.05$ over the admitted region; every argmin class frequency in $(0.05,0.95)$; rule-disagreement fraction $>0$; $0\in\mathcal{A}_{\rm sink}$ and every set before the query on 100 per cent of draws |
| **2** binds through the front door, the smoke test, the device certificate | V-14 / V-24: every plant fails at $O(1)$ as a FOUND cell with its own `kind` | the $0.9749 / 0.9165 / 1.000 / 0.4845$ pattern (`READ workdonenewseal.md:114-122`); `torch.equal` `True` at $\gamma=0$; cell price inside the band $[-1.8\%,+14.4\%]$ of $1.680$ s |
| **3** the arena at the design point | the realised paired sd fixes the MDE row; every arm is FOUND, not NAMED | realised paired sd against the pilot's $0.050146$; MDE$_8$ from the noncentral-$t$ bisection ($0.039827$ at sd $0.034451$, `RUN[VENUS]`, `RUN[MARS]` $0.0398266$) |
| **4** verdicts, the remaining certificates, the dossier | the five bets scored with their counters; K-E1 fires or escalates | per-bet PASS / KILL / SPLIT with the deciding interval; $\Lambda$ per seed against $2.7055$ and $\ln 4096 = 8.318$ |
| **5** the cost law, memory, kernels | no price is quoted at any $s>64$ before the sweep runs (K-9) | the fitted exponent's CI in $s$; the run-order Spearman $\rho$ against the record's $+0.7029$ |
| **6** the Kaggle gate and the LM cell | the author's explicit yes is a node, not a formality | every local certificate line filled; open rulings listed by name beside the request |

**DAG conventions.** Every card names its **prerequisites** and what it is **independent-of** (`CONTRACT.md` D-1: work is a DAG; parallel dispatch only on nodes with no shared repository state). A card marked **parallel-safe** touches no file another live card touches and may be picked up in any order the prerequisites allow; `results/*.jsonl` are append-only per lane (L-G2), so two cards writing different lanes are parallel-safe, two cards writing one lane are not. A card's **price** is tagged `[FITTED]` with its R-squared, `[RUN]`, `[ASSUMED]` with its reason, `DERIVED` with its arithmetic, or `NOT MEASURED` with the instrument it needs. **Evenings** are the author's time, never GPU time. `★` marks a cheapest-decisive card. Aliases on a header line name the planets' own card ids for the same content, so a reader of one roadmap can find its card here.

**The five cheapest decisive cards.** Each settles something alone, at $0$ GPU-s or seconds, and none blocks another (the coordinator's order, `judge/bind_ledger.md` §7):

1. **J-L18** — the ten `[M]` Lean targets build with their refusals and `#print axioms` ($0$ GPU-s). Decides whether the paper may write "machine-checked" for component (k).
2. **S-01** — the extended identity manifest with its one-unit drift plant ($0$ GPU-s). Decides whether any cell in the lane can be attributed at all (L-2: `0 of 24` attributable, `READ V20_R15_LEAP_LEDGER.md:23`).
3. **S-12** — the BED-S domain census on 512 draws ($0$ GPU-s). Decides admission, which is the record's dominant failure shape made free (D-4, M-3, V-8).
4. **S-20** — the ChaCAL-diag smoke test at $\gamma=0.9$ on BED-M with the B-J plant battery ($\approx 1.7$ s). Decides whether the solve path runs, is deterministic, and prices as predicted.
5. **S-61** — the $\gamma$-pinning likelihood-ratio instrument specified with the boundary null, the held-out set and the seed rule ($0$ GPU-s). Without it Bet C cannot be scored.

**FIRST EVENING — five cards that each settle something at $0$ GPU-s or seconds.** `J-L18` (with `J-L0`, `J-L1`, `J-L3` written the same sitting), `S-01`, `S-12` (with `S-11` written first), `S-20`, `S-61`. A sixth, independent of all five and long owed, is **S-52**: the $\approx 6$ GPU-s capped run at seeds 2, 3, 7 on BED-M that the record priced at it.7, it.8, it.9 and it.35 and never took (`READ V20_R15_IT35_JUPITER.md:157-159`). The evening's total GPU cost is under $20$ seconds.

**How to read a KILL.** A KILL line is the point estimate, not the tail. The record's only scored prediction census reads 9 checked, 9 adverse, 7 of 8 signed rows optimistic, one-sided sign test $p=0.0352$ (`READ V16_CALIBRATION.md:96-100`), so D-CALIB-1 makes the counter the planning estimate for every unscored card. That rule is why the plan is ordered cheapest-refutation-first and why every phase can end honestly.

---

## 5.2 The laws in force

One line each, with its source line (`sections/sec_state.md` §S.4), and the mechanism each pays for. These bind every card below.

| law | one line | source | pays for |
|---|---|---|---|
| **D-1** dependency | work is a DAG; parallel dispatch only on nodes with no shared repository state; git is the shared state | `READ CONTRACT.md:27-33` | the independent-of field on every card |
| **D-2** skills are modes | planet names label responsibilities inside documents, never concurrent processes | `READ CONTRACT.md:35-38` | the `J-/S-/M-/R-/V-/N-` prefixes here |
| **D-3** loop gate | the iteration count comes from the DAG's critical path, never chosen by an agent | `READ CONTRACT.md:40-44` | §5.4's evenings count |
| **D-4** order | contracts that schedule work behind an unreached round are staged, not started | `READ CONTRACT.md:55-57` | the phase-1 admission gate |
| **L-DOM** | every theorem that gates a run ships a domain census; no overlap means decoration | `READ CEQ_V16_CONTRACT.md:54-56` | V-25 (`READ MISTAKES.md:1954-1956`) |
| **L-SIGN** | a counter-prediction of equal specificity beside every prediction; a calibration column across rounds | `READ CEQ_V16_CONTRACT.md:58-61` | D-7 (`READ MISTAKES.md:2037-2039`) |
| **L-DIAG** | a contract prescribes what a diagnostic must distinguish, never which statistic does it | `READ CEQ_V16_CONTRACT.md:63-66` | M-18 (`READ MISTAKES.md:2103`) |
| **L-FLOOR** | every capability number ships beside its information floor, so "how good" reads as distance-to-floor | `READ CEQ_V20_R15_CONTRACT.md:60-63` | C15: `floor_1` is not a floor (`READ V20_R15_JOURNAL.md:51`) |
| **L-CERT** | every sparsity mask ships its certificate, F0 exact or F1 with $\delta$ printed; a mask without one is refused | `READ CEQ_V20_R15_CONTRACT.md:64-67` | V-10, V-17 |
| **L-EQ** | `[V]` is inadmissible for a load-bearing statement; `[V-eq]` needs the theorem, its hypotheses and one numeric instance | `READ CEQ_V15_CONTRACT.md:51-55` | P-10 |
| **L-LEAN** | the arm may not be trained before its identity theorems are green | `READ CEQ_V15_CONTRACT.md:57-58`; `READ CEQ_V16_CONTRACT.md:50-51` | P-11 |
| **L-FIRST** | predictions and counters are filed before each run — listed as standing; its definitional text was not located this session | `READ CEQ_V16_CONTRACT.md:49, :191` | M-2 |
| **L-G2** | journals never move and are never deleted; a superseded cell keeps a supersede marker | `READ V17K_RULINGS.md:62-64` | P-3 |
| **FOUND-not-NAMED** | a wing named rather than found is struck; FOUND iff `results/` holds a record with its `kind` | `READ CEQ_V20_R15_CONTRACT.md:49, :266` | P-4 |
| **R-SKY** | the native skyline is read beside every bed; "beats softmax" is not licensed where softmax is a fellow approximator | `READ CEQ_V16_CONTRACT.md:209` | `MISTAKES.md` D-1 |
| **Ruling 1** | determinism: CUDA with `warn_only=True`; bitwise for replay and every deciding forward cell; CPU-strict refused | `READ V17K_RULINGS.md:39-45` | M-10, M-16 |
| **Ruling 2 / 2a** | $\beta$ learnable and logged per instance; no sentence transfers across corners without a bind at the corner it describes | `READ V17K_RULINGS.md:47-54, :275-325` | V-22 |
| **Ruling 3** | matched parameters: a $0.032$ per cent residual is matched; exact counts in every table header; never re-architect to close it | `READ V17K_RULINGS.md:56-59` | M-8 |
| **Ruling 6** | the flight envelope 6a-6f: stale is two polls, tier precedence, OOM keyed per shape, deciding-cell list frozen at launch | `READ V17K_RULINGS.md:71-81` | V-15 |
| **Ruling 7** | a bed is carried as generator plus seed plus hash and regenerated in-notebook; the hash assertion is the provenance | `READ V17K_RULINGS.md:83-86` | P-1 |
| **Ruling 8** | Q2 dropped: `10.578 GiB` against the 4060's `7.996 GiB` | `READ V17K_RULINGS.md:328-339` | P-8 |
| **Ruling 9** | R1-prime cells fall under the measured-floor clause; no equality statement without its floor printed beside it | `READ V17K_RULINGS.md:354-379` | L-FLOOR |
| **Ruling 10-prime** | "pinned" by likelihood ratio on held-out, with the minimum detectable departure printed; supersedes the ulp criterion | `READ V17K_RULINGS.md:389-436` | M-20 |
| **the R15 kills** | named-not-found is struck; a mask without a certificate is refused; a capability number without its floor is not a number; a leap output acted on before its instance runs is struck | `READ CEQ_V20_R15_CONTRACT.md:265-273` | §5.9 |

**Two laws thinner than their citation, recorded so the plan does not lean on them.** `L-GRADE` (the F0-F4 rubric) is cited once and no rubric text exists in the tree (`READ V20_R15_THEORY_TABLE.md:22-24`); Rulings 11 and 12 are cited as binding and have no text (`RUN grep`, zero hits outside the two citing lines). This plan states the grading it uses and marks it a reconstruction (P-11).

**The author's supersession of 2026-09-03.** On that date the author licensed a new primitive — "rethink the shape" — and instructed "do not write any code only docs" (`READ BRIEF.md:11-30, :40-42`). The round-15 contract's clause *no new primitive that is not a repair* is therefore **superseded by the author for this document**, and this plan says so here rather than by editing the contract, which is verbatim-of-record and is never edited (`READ CEQ_V20_R15_CONTRACT.md:3-6`). What the supersession does **not** lift, because the author did not lift it: the leap's remaining clauses (named theorems with runnable instances, predictions **and** counters, every untested claim with its cheapest killer, instances RUN before anything is acted on) and every standing law in the table above. What it changes in kind: the iteration count of this programme is derived from its own DAG's critical path (§5.4), never inherited from the 45 iterations the R15 schedule named; and the Phase-E template — candidate built with identity binds, manifests, certificates, parameter-matched, full arena, verdict by the registered clauses, purge, prognosis (`READ CEQ_V20_R15_CONTRACT.md:153-169`) — is inherited with the beds and labels replaced. Mechanism: P-3 (a stale claim never retracted) — the supersession is dated, and §5.11's corrections index is where it is amended if the author changes it.

---

## 5.3 The phases and their milestone cards

Card format, every field on every card: **id · title** — phase · prerequisites · independent-of · parallel-safe · aliases; then *build* (a specification, never code), *prove*, *measure*, *PASS*, *KILL* with what dies, *price*, *mechanism*, *deliverable*, *evenings*.

---

### PHASE 0 — free work: theorems, manifests, specifications, censuses

**Goal.** Put every identity the paper leans on into Lean or into a written derivation, give the lane an identity manifest that refuses an undeclared field, and freeze the specifications and thresholds later phases may not move (M-2). Price of the whole phase: $0$ GPU-s.

**The gate.** L-LEAN. `lake build` from `lean/` exits `0`, zero `sorry`, zero `sorryAx`, axioms `[propext, Classical.choice, Quot.sound]` only, every `[M]` card green; a target that does not build is `[S]` and is never cited as proved (K-K). The manifest's `5/5` single-field flips move `manifest_hash` and an omitted declared field raises. Every obstruction sentence carries its vacuity inequality: $h\,m\,p=512\ge64$, $n\log_2 n=384<544$, $64^{1/16}=1.2968$, all vacuous at $s=64$, $d=16$, $p=32$. The author decides from three printed things: the build log, the flip count, the census table. **No arm trains until this gate is green.**

**J-L0 ★ · Lay the `Shape` module and run the Mathlib name census** — phase 0 · prereq none · independent-of every card · parallel-safe · aliases none
*build* one import root `lean/CEQ/Shape.lean` over `CEQ.Contraction`, `Occupancy`, `Nilpotent`, `V16Domain`, `OracleSeparation`, with files `Shape/{Parity,Triangular,Boundary,Segment,Displacement,Neumann,Committor}.lean`, none above the `V15Source.lean` scale (201 lines, 7 declarations); shared definitions stated once — `LowerTri`, `Absorbing P A`, `IsSink 0`; journal fields per card: declaration, `file:line`, grade, `#print axioms`, `lake build` exit, Mathlib rev, census line.
*prove* nothing; the census is a `grep` over the vendored tree at rev `a45ae637` (2024-04-04, toolchain `leanprover/lean4:v4.7.0`, `READ lean/lean-toolchain`) for the eight Mathlib names the `[M]` cards rest on, each recorded `[V-name]` with `file:line` or `[U]`.
*measure* `lake build` exit `0` on the empty module (the warm cache reads `[1530/1531]`, `READ workdonenewseal.md:41-45`).
*PASS* all eight names resolve at the pinned rev (they do, `plan_jupiter.md` §0). *KILL* a missing name demotes its dependent card to `[S]`, which must then supply the lemma itself; what dies is the `[M]` grade, never the identity.
*price* $0$ GPU-s · *mechanism* P-11, P-4 · *deliverable* `docs/LEAN_SHAPE_MANIFEST.md` · *evenings* 1

**J-L1 ★ · Parity at $\gamma=0$, with its refusal** — phase 0 · prereq J-L0 · independent-of J-L2 to J-L17 · parallel-safe
*build* `gamma_zero_is_softmax` ($P\,(1-0\cdot P)^{-1}V=PV$) and the witness `gamma_half_is_not_softmax` on `Fin 2`.
*prove* `[M]`; `Matrix.inv_one`, `zero_smul`, `sub_zero`; the witness by `decide` or `norm_num`.
*measure* the float instance stays RUN: `torch.equal(O(0), PV) = True` at $s=64$, $d=16$ (`RUN[coord]`); rejection region $2.3002850040264393$ at $\gamma=0.5$; the record's own softmax corner is $1.110223\times10^{-16}$ off `ceq/lm.py` on `19/64` entries (`READ V16_ARM_SMPRIME.md:266-293`), so "bitwise" is against the lane's `softmaxAttn`, never a fused kernel.
*PASS* both declarations build with the three-axiom set. *KILL* the witness does not close: the parity proposition ships RUN-only and Proposition 1 loses its Lean rejection region.
*price* $0$ GPU-s · *mechanism* V-3 (the pass is by construction, so the refusal is mandatory), V-24 · *deliverable* `lean/CEQ/Shape/Parity.lean` · *evenings* 1

**J-L2 · Row 0 of a causal softmax is absorbing at $\beta=1$** — phase 0 · prereq J-L0 · independent-of all but J-L10, J-L18 · parallel-safe
*build* on the tree's `Hop` (`READ V16Domain.lean:366-378`): `bos_row_is_absorbing`, refusal `bos_row_not_absorbing_at_beta_zero` (at $\beta=0$ row 0 reads $e^{qk_{00}}=2.0138$, `RUN[M]`).
*prove* `[M]`; `Finset.sum_range_one`, `Real.exp_ne_zero`, `Real.rpow_one`, `div_self`. This ships fact F1 of `THESIS_CORRECTIONS_2.md` §1 as a theorem.
*measure* BOS undeclared gives $\rho(Q)=1.000000$, $\det(I-Q)=0.0$; declared as $\mathcal{A}_{\rm sink}=\{0\}$ gives $\rho(Q)=0.692660$ (`RUN[P]`, `RUN[J]`).
*PASS* builds, and the domain census reads $100$ per cent of BED-S draws with $0\in\mathcal{A}_{\rm sink}$. *KILL* the theorem needs $0<Z_{\rm norm}$ as an extra hypothesis the tree does not supply: grade `[S]` and census line X-2 stays a per-batch RUN.
*price* $0$ GPU-s · *mechanism* V-25, V-12 · *deliverable* `lean/CEQ/Shape/Boundary.lean` · *evenings* 1

**J-L3 ★ · $I-\gamma P$ is a unit on the causal class, with the diagonal interval** — phase 0 · prereq J-L0 · independent-of J-L1, J-L2, J-L5, J-L8, J-L9, J-L11, J-L12 · parallel-safe
*build* `lower_triangular_isUnit`, `diag_one_sub_smul_pos` (diagonal of $1-\gamma P$ in $[1-\gamma,1]$; the open upper bound needs $0<P_{ii}$, which J-L5 supplies), refusal `zero_diag_not_unit`.
*prove* `[M]` — `Matrix.det_of_lowerTriangular` (`Block.lean:265`) and `Matrix.isUnit_iff_isUnit_det` (`NonsingularInverse.lean:151`), both confirmed at the pinned rev.
*measure* minimum diagonal $0.4$ at $\gamma=0.6$, attained by row 0 and every absorbing row (`RUN[M]`); `solve_triangular` against the dense inverse $1.7763568394002505\times10^{-15}$ (`RUN[coord]`).
*PASS* builds. *KILL* the `LinearOrder (Fin n)` instance clashes: prove the determinant by induction, grade `[S]`, plus one evening; the triangular branch stalls one evening and the substitution stands as RUN.
*price* $0$ GPU-s · *mechanism* M-8 (the shape is priced as a substitution; this theorem is what makes the substitution well-defined) · *deliverable* `lean/CEQ/Shape/Triangular.lean` · *evenings* 1

**J-L4 · Consequences propagate forward only** — phase 0 · prereq J-L3 · independent-of J-L5 to J-L9, J-L11, J-L12 · parallel-safe
*build* `inv_lowerTri`, `later_boundary_unreachable` (F2: a set after the query has committor exactly zero), `causal_forward_only`, refusal `dense_P_displaces_backward`.
*prove* `[M]` via `Matrix.blockTriangular_inv_of_blockTriangular` (`Block.lean:346`); `causal_forward_only` is one line routed through `inv_lowerTri` rather than Sherman-Morrison.
*measure* $\max|\Delta z_{<12}|=0.0$ on the row clamp at $i=12$ (`RUN[P]`); $5.6\times10^{-17}$ on the non-nilpotent softmax corner (`RUN[F]`); the triangularity plant (non-causal $P$) reads $0.0761$ and $0.0868$ (`RUN[M]`).
*PASS* builds. *KILL* the `IsUnit`-to-`Invertible` cast does not elaborate: grade `[S]` and prove `inv_lowerTri` by J-L6's block formula; the forward-only claim stays a RUN with its plant.
*price* $0$ GPU-s · *mechanism* V-8, V-24 · *deliverable* `lean/CEQ/Shape/Triangular.lean` · *evenings* 1

**J-L5 · The softmax corner is not nilpotent (the regime boundary)** — phase 0 · prereq J-L0 · independent-of everything but J-L18 · parallel-safe
*build* `lowerTri_pow_diag`, `softmax_corner_not_nilpotent`; the refusal is already in the tree (`Nilpotent.pow_card_eq_zero :77`, `one_not_nilpotent :105`).
*prove* `[M]`; `Matrix.mul_apply`, `Finset.sum_eq_single`, `pow_pos`.
*measure* $\max_{i\ge1}(\gamma P)^{32}_{ii}=6.27\times10^{-13}$ beside $\gamma^{32}=7.96\times10^{-8}$ attained at $(0,0)$ (`RUN[M]`) — the discriminating pair the judge fixed after a V-4.
*PASS* builds. *KILL* none foreseeable; if the `Fin` order must be unfolded, plus one evening. Nothing measurable dies: the RUN pair stands and the default "keep the diagonal, carry the Neumann certificate" is unaffected.
*price* $0$ GPU-s · *mechanism* V-25, P-3 · *deliverable* `lean/CEQ/Shape/Triangular.lean` · *evenings* 1

**J-L6 ★ · A zero gate splits the resolvent into blocks** — phase 0 · prereq J-L0 · independent-of J-L1 to J-L5, J-L8 to J-L17 · parallel-safe
*build* `segmentation_blockdiag` on `StrictlyLower`, `resolvent_fromBlocks`, the chain corollary `chain_zero_gate_cuts` `[S]`, refusal `tiny_gate_does_not_cut` at a $-30$ logit.
*prove* `[M]` for the first two (`inv_fromBlocks_zero_21_of_isUnit_iff`, `SchurComplement.lean:202`, confirmed); the induction is `pow_entry_zero`'s.
*measure* segmentation zeros `array_equal` on the gated corner (`RUN[M]`); the $-30$ plant ($e^{-30}=9.36\times10^{-14}$) leaves the block non-zero; the $10^{-300}$ plant is struck as float64 underflow. Domain census: BED-M admits a zero gate on `3 of 3` values (`READ V16Domain.lean:304`); BED-S at the softmax corner reads `0 of N` and the theorem is silent there.
*PASS* builds. *KILL* `fromBlocks`' `Sum m n'` indexing does not match the `Fin n` cut: ship the entrywise statement only and demote the block inverse to a remark; the F0 row keeps its `torch.equal` RUN and no `[M]` tag.
*price* $0$ GPU-s · *mechanism* L-CERT, P-11, V-25, V-2 · *deliverable* `lean/CEQ/Shape/Segment.lean` · *evenings* 1

**J-L7 · A cut makes the segment head absorbing and severs the sets behind it** — phase 0 · prereq J-L6, J-L2 · independent-of J-L8 to J-L17 · parallel-safe
*build* `cut_makes_segment_head_absorbing`, `cut_severs_boundary_sets`, on a gated hop `HopGated` defined once with the product taken directly (the tree's `Hop` carries no zero, `no_prefix_scan_represents_a_zero_gate :165`).
*prove* `[M]` both; objects as J-L2 and J-L6.
*measure* after a cut at $c$: $P_{cc}=1.0$, $\rho(Q)=1.0$ with $c\notin\mathcal{A}$ (`RUN[J]`); once $c$ is declared, $q^{(0)}=q^{(1)}=0.0$ past the cut (`RUN[M]`).
*PASS* builds. *KILL* defining `HopGated` forces re-proving `three_corners_containment` for it: plus one evening, still `[M]`; if abandoned, segmentation and the committor channel are kept apart by rule rather than by theorem.
*price* $0$ GPU-s · *mechanism* V-25, D-3, V-8 · *deliverable* `lean/CEQ/Shape/Segment.lean` · *evenings* 1

**J-L8 ★ · The displacement identity with its planted zero** — phase 0 · prereq J-L0 · independent-of everything but J-L14, J-L18 · parallel-safe
*build* `displacement_identity` and refusal `const_value_zero_displacement`.
*prove* `[M]`; `Matrix.sub_mulVec`, `smul_mulVec_assoc`, `mulVec_sub`; four lines.
*measure* residual $2.897\times10^{-16}$ (`RUN[P]`), $1.2\times10^{-15}$ (`RUN[I]`), $1.36\times10^{-15}$ (`RUN[F]`), $1.03\times10^{-15}$ (`RUN[J]`); $V\equiv\mathbf 1$ gives $\max|\Delta z|\le10^{-15}$ (`RUN[F]` $1.1\times10^{-16}$; a bitwise zero is a one-code-path accident); Gaussian $V$ gives $0.1096$, $0.363$, $1.127$ on three draws — the rejection region.
*PASS* builds. *KILL* none foreseeable; nothing dies, the three independent RUNs stand.
*price* $0$ GPU-s · *mechanism* V-24, D-5 · *deliverable* `lean/CEQ/Shape/Displacement.lean` · *evenings* 1

**J-L9 · A causal operator is never the oracle's undirected chain (D-2 in regime S)** — phase 0 · prereq J-L0 · independent-of everything but J-L18 · parallel-safe
*build* `lowerTriangular_ne_symmSupport`, extending `OracleSeparation.oracle_ne_resolvent` (`:166`, `StrictlyLower` only) to the diagonal-kept class; refusal pattern `zero_not_a_counterexample` (`OracleSeparation.lean:193`).
*prove* `[M]`, one line: `SymmSupport` gives $0<Q_{ji}$, `LowerTri` gives $P_{ji}=0$.
*measure* BED-1's chain instantiates the hypotheses: `Nonneg True`, `SymmSupport True`, $\rho(Q)=0.9408612510154677$, $Q^{11}\ne0$ (`RUN[coord]`).
*PASS* builds. *KILL* none. If it fails, the ruling that BED-S's environment must be a DAG in token order stands on the RUN alone rather than on a theorem.
*price* $0$ GPU-s · *mechanism* D-2 (`READ MISTAKES.md:710`), V-25 · *deliverable* `lean/CEQ/Shape/Committor.lean` · *evenings* 1

**J-L10 · The committor read and its causal invertibility** — phase 0 · prereq J-L3, J-L2 · independent-of J-L5 to J-L9, J-L11, J-L12 · parallel-safe
*build* `committor_is_resolvent_read` (a), `isUnit_one_sub_transient_causal` (b-causal), `transient_diag_lt_one_of_bos_declared`, refusal `bos_undeclared_singular`.
*prove* `[M]` all three; the last is the positivity argument $P_{ii}=1-\sum_{j<i}P_{ij}<1$ once $P_{i0}>0$.
*measure* $\rho(Q)=\max_{i\in T}P_{ii}=0.692660$ (`RUN[P]`); BED-1's real sets $A=[0]$, $B=[1]$, $|T|=9$: resolvent read against `bed["q"]` $0.0$, harmonic residual $1.0408340855860843\times10^{-17}$ (`RUN[coord]`); the must-fire perturbation drives the residual to $10^{-6}$, ratio $9.6\times10^{10}$.
*PASS* builds. *KILL* the strict-positivity hypothesis fails on gated corners (a zero gate zeroes $P_{i0}$): the theorem is stated for the softmax corner only, census line X-3 stays a RUN on gated corners, and the head ships on a per-batch determinant check.
*price* $0$ GPU-s · *mechanism* D-2, V-10, V-16 · *deliverable* `lean/CEQ/Shape/Committor.lean` · *evenings* 1

**J-L11 · Corner 3 is the sub-diagonal resolvent, entrywise** — phase 0 · prereq J-L0 · independent-of everything but J-L6's chain corollary and J-L18 · parallel-safe
*build* `subdiag`, `subdiag_strictlyLower`, `subdiag_pow_entry`, `pathprod_is_chain_resolvent`.
*prove* `[S]`; the induction is `pow_entry_zero`'s with one surviving term; objects `Finset.sum_eq_single`, `Finset.prod_Ico_succ_top`. The $n=3$ instance is in the tree (`V15Source.forward_map_fills_in :168`).
*measure* $\max|G-(I-A)^{-1}|=0.0$; last row against `equilibrium_oracle` $6.217248937900877\times10^{-15}$; $A^{64}=0$ exactly (`RUN[coord]`).
*PASS* builds. *KILL* the `Ico` product bookkeeping does not close: split the two cases, plus one evening; the containment stays a $0.0$ RUN and `shape - corner-3` on BED-M remains VOID by registration rather than VOID by theorem.
*price* $0$ GPU-s · *mechanism* D-2, `MISTAKES.md` D-1 · *deliverable* `lean/CEQ/Shape/Chain.lean` · *evenings* 2

**J-L12 · The Neumann certificate, its attainment, and the mask amplification** — phase 0 · prereq J-L0 · independent-of J-L1 to J-L11 · parallel-safe
*build* pointwise in the style of `Contraction.lean`, no norm instance: `resolvent_sup_bound`, `isUnit_one_sub_smul`, `neumann_truncation_bound`, `neumann_tail_attained`, `mask_amplification` `[S]`, refusal `nonstochastic_breaks_bound` at a **convergent** $\gamma_0=3/5$ with rows $3/2$.
*prove* `[S]` all; the $K$-fold iterate of `weighted_contraction` is the only induction.
*measure* equality to the printed digits at $(\gamma,K)=(0.6,2)$: $0.540000$; $(0.7,4)$: $0.5602333$; $(0.7,16)$: $7.754350\times10^{-3}$ (`RUN[P]`); the convergent plant reads $7.2900$ against bound $0.5400$; the coordinator's $\gamma=0.7$ plant ($119.37$ against $1.143$) is a divergent partial sum ($\rho(\gamma P)=1.05$) and is labelled so; the mask witness at $s=3$, $\gamma=0.9$, $\varepsilon=0.1$ reads $0.5263157894736843$ against the naive $0.1$ (`RUN[J]`).
*PASS* builds, and the paper prints $\delta\,\|V\|_\infty$, never a bare $\delta$, with $1/(1-\hat\gamma)$ beside it. *KILL* `RowStochastic (P^k)` is missing as a lemma: plus one evening; if the amplification constant does not close, ship the $s=3$ witness as `[M]` and the general bound as `[D]`. If the card dies entirely, every $\delta$ is printed from the textbook (`meyer-2000-matrix`, `horn-2013-matrix`) with the RUN plant and no arena cell is lost.
*price* $0$ GPU-s · *mechanism* V-3, V-10, V-24, V-17, L-CERT · *deliverable* `lean/CEQ/Shape/Neumann.lean` · *evenings* 2

**J-L13 · The resolvent is one forward substitution, and $\Pi_\gamma$ is a mixture** — phase 0 · prereq J-L3, J-L12 · independent-of J-L5 to J-L9, J-L11 · parallel-safe
*build* `fwdSub`, `resolvent_is_triangular_solve`, `mixing_matrix_rowStochastic`, refusal-shaped `bare_read_row_sum` $=1/(1-\gamma)$.
*prove* `[S]` all three; non-negativity of the inverse follows from $(1-\gamma P)^{-1}=\mathrm{occupancy}(\gamma P,N)+(1-\gamma P)^{-1}(\gamma P)^N$ with both terms non-negative for every $N$.
*measure* $1.7763568394002505\times10^{-15}$ solve against dense inverse (`RUN[coord]`); $\Pi_\gamma$ row sums $1.000000000000000$ with and without absorbing rows (`RUN[F]`); bare row sums $2.5=1/(1-0.6)$ (`RUN[J]`, `RUN[M]`); support of $\Pi_\gamma$ equals support of $P$ on a dense causal softmax (`RUN[J]`).
*PASS* builds. *KILL* the well-founded recursion does not elaborate: state the characterisation $M_{ii}z_i=v_i-\sum_{j<i}M_{ij}z_j$ instead (`[M]`) and drop the executable definition; the cost law then stands on the RUN.
*price* $0$ GPU-s · *mechanism* M-8, V-17, V-23 · *deliverable* `lean/CEQ/Shape/Triangular.lean`, `Shape/Neumann.lean` · *evenings* 2

**J-L14 · The suffix re-solve and the rank-one row clamp** — phase 0 · prereq J-L4, J-L8 · independent-of J-L5 to J-L7, J-L9, J-L11 to J-L13, J-L15 to J-L17 · parallel-safe
*build* `suffix_resolve_eq_full`, `sherman_morrison_row` (docstring: a row clamp only; a token rewrite is rank $s-i$), `sm_denominator_pos`, refusal `dense_P_breaks_suffix_resolve`.
*prove* `[S]`; the algebra is `sherman-1950-inverse-adjustment` and `hager-1989-updating` `[V]` re-proved on `Fin n`; the denominator equals $(1-\gamma p'_{ii})/(1-\gamma P_{ii})>0$ for every $\gamma<1$.
*measure* suffix against full $5.6\times10^{-17}$ on the non-nilpotent corner (`RUN[F]`); closed form against re-solve $1.2339847026143769\times10^{-15}$ with denominator $1.9$ (`RUN[I]`); $0.511918$ (`RUN[F]`); $m$ columns cost $m/d=0.5$ of one solve at $(8,16)$, SPLIT band $(0.5\times, 8\times)$ filed.
*PASS* builds. *KILL* the `Matrix.of` row update does not simp: restate with `Matrix.vecMulVec`, same content; if abandoned, the move price is a DERIVED band with the RUN identities.
*price* $0$ GPU-s · *mechanism* M-8, P-8, V-24, D-5 · *deliverable* `lean/CEQ/Shape/Displacement.lean` · *evenings* 1

**J-L15 · The reach-avoid conservation row and the degeneracy lemma** — phase 0 · prereq J-L10 · independent-of J-L5 to J-L9, J-L11 to J-L14 · parallel-safe
*build* `reach_avoid_sum_one` (hypothesis: the sets exhaust the absorbing states), `no_goal_no_sink_forces_max_ge_inv_K`, `isUnit_one_sub_of_perron` (b-chain, the oracle's non-causal $Q$ only), refusal `unit_needs_rho_lt_one`.
*prove* `[S]`; `kemeny-1976-finite` and `grinstead-1997-probability` Thm 11.6 `[U]` own the identity $B=NR$ — the card machine-checks the packaging.
*measure* the four reads sum on $T$ to $[1.000000000000000, 1.000000000000001]$ (`RUN[P]`); with BOS inside a constraint set and no goal, the two committors sum to $1.000000000000000$ and $\min_T\max_k q^{(k)}=0.548718\ge1/2$ — the lemma's instance. The coordinator's $0.605$ was computed with the goal counted and is not this lemma's instance.
*PASS* builds. *KILL* `Finset` submatrix indexing does not close in one file: state on the full `Fin n` with `Absorbing` rows, plus one evening; if abandoned, the row is printed with its residual on every batch (V-23) and the lemma is DERIVED.
*price* $0$ GPU-s · *mechanism* V-12, V-8, V-23, V-25 · *deliverable* `lean/CEQ/Shape/Committor.lean` · *evenings* 2

**J-L16 · The masked-sink finite sum, and BED-1's solve as the committor read** — phase 0 · prereq J-L12, J-L10 · independent-of J-L5 to J-L9, J-L11, J-L14 · parallel-safe
*build* `masked_sink_finite_sum` ($V_0=0$), refusal `goal_channel_breaks_finite_sum`, `bed1_committor_eq` unpacking `READ ceq/beds/bed_1.py:173-174, :188-198`.
*prove* `[S]` both.
*measure* the masked route reads $4.4\times10^{-16}$ on a value-zero channel against $1.99\times10^{-7}$ on $\mathbf 1_{\mathcal{A}_0}$ with $0\in\mathcal{A}_0$ (`RUN[M]`); BED-1's harmonic residual $1.0408340855860843\times10^{-17}$ at $T=0.25$, jitter $0.05$, seed 11.
*PASS* builds. *KILL* the unpacking needs the bed's kernel as a Lean object: state it for an abstract diagonally dominant $L$ (`[S]`); if abandoned, the paper keeps the diagonal and Proposition 4 everywhere, which is already the default.
*price* $0$ GPU-s · *mechanism* V-25, V-4, D-2 · *deliverable* `lean/CEQ/Shape/Committor.lean` · *evenings* 1

**J-L17 · Register the two deferred targets, statements only** — phase 0 in place, phase 5 in grade · prereq J-L0 · independent-of all · parallel-safe
*build* `Shape/Deferred.lean`, **not** imported by `CEQ.lean` so the root build stays `sorry`-free: `hitting_time_transform` and `f1_cantelli_union`, both `[D]` — this Mathlib has no finite-chain hitting-time object (`RUN[P] grep`, zero files; the stopping-time machinery at `Mathlib/Probability/Martingale/Basic.lean:508-510` is for filtrations of a measure space).
*prove* nothing this round; the content is J-D3.
*measure* read against $E[\gamma^{\tau-1}]$ to $8.33\times10^{-17}$, Monte-Carlo-free, by the finite-sum identity (`RUN[J]`, `RUN[M]`).
*PASS* the file's statements type-check. *KILL* a statement needs an object the tree cannot name: write it in J-D3's finite-sum form; what dies is the Lean statement, never the derivation.
*price* $0$ GPU-s · *mechanism* P-11, P-4 · *deliverable* `lean/CEQ/Shape/Deferred.lean` · *evenings* 1

**J-L18 ★ · The build gate, the axiom print and the domain census** — phase 0 gate · prereq every `[M]` card it certifies (J-L1 to J-L10 at minimum) · independent-of the `[S]` cards it does not list · **not** parallel-safe · aliases M-0.1, M-0.4, R-01, V-2
*build* (i) `lake build` from `lean/` with the new imports in `CEQ.lean`; (ii) a `#print axioms` block per declaration in the pattern of `V16Domain.lean:586-594`; (iii) a domain census file in the `bedM_overlap_*` style (`READ V16Domain.lean:302-310`) giving, per theorem, the corpus and the fraction of it satisfying the hypotheses: `gamma_zero_is_softmax` `3/3`; `bos_row_is_absorbing` `3/3` at $\beta=1$, `0/3` at $\beta=0$; `segmentation_blockdiag` `3/3` on BED-M, `0/N` on BED-S's softmax corner (silent); `softmax_corner_not_nilpotent` `0/3` on BED-M, `N/N` on BED-S; `lowerTriangular_ne_symmSupport` `1/1` on BED-1; `neumann_*` `N/N` at $\hat\beta=1$, `0/N` at $\hat\beta\ne1$ (rows sum $1.31$ to $10.29$, `READ V16_ARM_SMPRIME.md:28-32`).
*prove* the census rows by `decide` where the value set is finite, by a printed RUN where it is a drawn batch.
*measure* `lake build` exit code; `grep -c '^theorem\|^lemma'` on the new files beside the record's `169` over 12 files; the axiom list per declaration.
*PASS* exit `0`, zero `sorry`, zero `sorryAx`, the three-axiom set, every `[M]` card green; a census line of $0$ per cent demotes that theorem to decoration on that bed (L-DOM). *KILL* (K-K) any `[M]` target not building is `[S]` and is never cited as proved; what dies is the paper's "machine-checked" sentence for component (k), narrowed to the rows that build — and **no arm trains until this reads green**.
*price* $0$ GPU-s (CPU minutes) · *mechanism* P-11, L-LEAN, L-DOM, V-25, V-16 · *deliverable* `docs/LEAN_SHAPE_CENSUS.md`, `lean/CEQ/Shape/Census.lean` · *evenings* 1

**J-D1 · The EMC remark, with both halves and the ruling** — phase 0 · prereq J-L0's notation · independent-of every card · parallel-safe
*build* a remark, never a proposition (`judge/bind_ledger.md` B-EMC is KILL-deleted). The half that holds: for $\gamma<1$ and any row-stochastic $P'$, cyclic or not, $x\mapsto V+\gamma P'x$ is a $\gamma$-contraction in $\|\cdot\|_\infty$, so the post-intervention fixed point is unique and "settle then intervene" equals "intervene then solve". The half where Dash applies: `dash-2005-emc` Thm 1 concerns a reduced model whose equilibrated form hides feedback; an explicit linear fixed point has no such gap; the case where it returns is $P'$ recomputed from $z$, a DEQ (`bai-2019-deq`), out of scope. What survives as a bind is J-L4's triangularity plant. The external argument for the re-solve is `momennejad-2017-sr` and `russek-2017-predictive` (mechanism `[U]`).
*prove* uniqueness in four lines from the contraction (DERIVED).
*measure* $8.9\times10^{-16}$ on a dense feedback $P$ (`RUN[J]`), $1.33\times10^{-15}$ (`RUN[I]`), $4.4\times10^{-16}$ (`RUN[F]`).
*PASS* the remark carries the three numbers and no proposition number. *KILL* a planted feedback instance reading above $0.1$ on the shape's class — impossible for $\gamma<1$; if it ever reads so, the algorithm is wrong, as the design's "violation" was (a suffix-only solve on a non-triangular $P$).
*price* $0$ GPU-s · *mechanism* V-24, D-7, V-3 · *deliverable* `docs/CEQ_SHAPE.md` §4.2 Remark · *evenings* 1

**J-D2 · Derive reach-avoid sum-to-one with a sink, and the degeneracy lemma** — phase 0 · prereq J-D1's notation; states J-L15 · independent-of all Lean cards · parallel-safe
*build* Proposition 8's clauses in full: (i) with $\mathcal{A}=\mathcal{A}_{\rm sink}\sqcup\mathcal{A}_0\sqcup\bigsqcup_k\mathcal{A}_k$ exhausting the absorbing states and $\rho(Q)<1$, $\sum_\bullet q^{(\bullet)}=\mathbf 1$ on $T$, because $\sum_\bullet R_\bullet\mathbf 1=\mathbf 1-Q\mathbf 1$ and $N(I-Q)\mathbf 1=\mathbf 1$; (ii) if the constraint sets alone exhaust absorption, $\max_k q^{(k)}\ge1/K$ by pigeonhole; (iii) the discounted reads sum below one, so the delay share $1-\sum_k E[\gamma^{\tau-1}\mathbf 1_k]$ is printed beside every safest-move reading. Owners cited before the object: `kemeny-1976-finite`, `grinstead-1997-probability` Thm 11.6 `[U]`, `summers-2010-reach-avoid`, `vanmoffaert-2013-chebyshev`, `fisac-2019-bridging`; `misra-2023-safety-constrained-mdp` carried to Limits.
*prove* DERIVED, steps as above.
*measure* the read-form sums $[0.227, 0.567]$ at $\gamma=0.6$ with the state-form/read-form ratio exactly $\gamma$ (`RUN[I]`); the conservation row $[0.9999999999999993, 1.0]$.
*PASS* census lines X-5, X-6, X-7 read the printed row on every batch. *KILL* a batch failing by more than $10^{-12}$ means a set was left undeclared, and the solve must raise rather than return (V-16).
*price* $0$ GPU-s · *mechanism* V-12, V-8, V-23, V-25 · *deliverable* `docs/CEQ_SHAPE.md` §4.2 Prop. 8 · *evenings* 1

**J-D3 · Derive the $E[\gamma^\tau]$ identity without a probability space** — phase 0 · prereq J-L10's statement · independent-of all · parallel-safe
*build* Proposition 3 in a finite-sum form the Lean tree can eventually carry: for $i\in T$ and $V=\mathbf 1_{\mathcal{A}_k}$, $z_i=\sum_{t\ge0}\gamma^t(P^t\mathbf 1_{\mathcal{A}_k})_i$; with $\mathcal{A}_k$ absorbing, $(P^t\mathbf 1_{\mathcal{A}_k})_i=\sum_{u\le t}f_i(u)$ with $f_i(u)\ge0$ the first-passage mass at step $u$; exchanging sums gives $(1-\gamma)z_i=\sum_u\gamma^u f_i(u)$, the read is $O_i=\sum_u\gamma^{u-1}f_i(u)$ by first-step analysis, and $\lim_{\gamma\uparrow1}O_i=q^{(k)}_i$ by monotone convergence of a non-negative series. The gap line: $q_i-E_i[\gamma^\tau\mathbf 1_k]=E_i[(1-\gamma^\tau)\mathbf 1_k]\ge(1-\gamma)q_i$, small where $q_i$ is. At $i\in\mathcal{A}_k$ the read is $1$, not $\gamma^{-1}$.
*prove* DERIVED; this overrules `refute_theory_math` item 8's universal $\ge1-\gamma$ gap.
*measure* $\max_T=1.42\times10^{-7}$ at $\gamma=1-10^{-6}$ on channel $\{15\}$ (`RUN[J]`); read against $E[\gamma^{\tau-1}]$ to $8.33\times10^{-17}$; BED-1's real sets $0.0$.
*PASS* the derivation carries the $(1-\gamma)q_i$ line and both RUNs. *KILL* none; the identity is classical and its owners are cited before the object is named.
*price* $0$ GPU-s · *mechanism* D-2, V-25, P-10 · *deliverable* `docs/CEQ_SHAPE.md` §4.2 Prop. 3 · *evenings* 1

**J-D4 · The DET-class statement, and what "depth-1" means** — phase 0 · prereq none · independent-of all · parallel-safe
*build* the displayed statement: the exact committor solves $(I-Q)q=R\mathbf 1$, the exact $z$ is $(I-\gamma P)^{-1}V$; inversion and iterated product are DET-complete with $\mathrm{NL}\subseteq\mathrm{DET}\subseteq\mathrm{NC}^2$ (`cook-1985-taxonomy` `[V]`); a triangular inverse is DET-hard (J-L11's `subdiag_pow_entry` is that embedding with scalar blocks); rational linear **equalities** are in DET and the P-complete problem is linear **inequalities**. So the shape relocates the log-depth from the parameter stack into the solve: $O(s)$ sequential rounds by substitution, or recursive block inversion at $O(\log^2 s)$ circuit depth and $O(\log s)$ matmul rounds with $O(s^3)$-class work — `NOT MEASURED, needs a parallel-prefix kernel timing`. "Depth-1" means one attention parameter set, $(P,\gamma)$; the circuit-depth reading is disclaimed in the same paragraph. Beside it the skyline facts and their vacuity inequalities.
*prove* DERIVED.
*measure* $64^{1/16}=1.2968$ against $Hdp=512$; $512\ge64$; $384<544$.
*PASS* every obstruction sentence carries its theorem number, model class and inequality (census X-15). *KILL* (K-10) any "cannot" sentence outside the skyline table is struck at assembly.
*price* $0$ GPU-s · *mechanism* P-8, V-17, P-10, V-25, P-3 · *deliverable* `docs/CEQ_SHAPE.md` §5.3 · *evenings* 1

**J-D6 · Restate the Cheeger line on a symmetrised surrogate** — phase 0 · prereq none · independent-of all · parallel-safe
*build* `MATHEMATICS.md:455` cites `levin-2017-markov-mixing` Thm 13.10, whose bound holds for a
**reversible** $P$; the shape's causal $P$ is not reversible (a lower-triangular $P$ with $P_{i0}>0$ has $P_{0i}=0$), so the line is V-25 as written. The restatement: the honest surrogate is the transient block's lazy symmetrisation $Q_s=(Q+Q^\top)/2$ on $T$, whose Cheeger bound controls the surrogate's gap and **not** $\rho(Q)$; the quantity the hop ladder truncates is $\rho(Q_{\rm env})$, read off the ladder and never predicted from a spectrum ($0.9964$ against $0.9985$ on `LargestJoin_S2Rips_1024`, `READ PRIOR_ART.md:703-707`); on the DAG substrate $Q$ is lower-triangular with $\rho(Q)=\max_T P_{ii}$, so there is no bottleneck spectrum to bound and the Cheeger line applies to the oracle cross-check chains (BED-1, E4-prime) only.
*prove* DERIVED; the reversibility hypothesis is quoted once, under fifteen words.
*measure* the E4-prime bottleneck reading $1372.50$ is that bed's own constant, marked V-22, never a BED-S bar (`READ scale/e4_harmonic.py:247-263`).
*PASS* the restated line names the surrogate, the object bounded and the bed it applies to. *KILL* any $g\le2\phi$ on a causal $P$ without a stated symmetrisation is struck (K-10 type).
*price* $0$ GPU-s · *mechanism* V-25, P-10, V-17, V-22 · *deliverable* `docs/CEQ_SHAPE.md` §4.4 and a `MATHEMATICS.md` §7 correction pointer · *evenings* 1

**J-D7 · The mask-amplification and certificate-units derivation** — phase 0 · prereq J-L12's statement · independent-of all · parallel-safe
*build* Proposition 4(ii) in full with the $s=3$ closed-form instance ($\gamma=0.9$, $\varepsilon=0.1$, $V=e_0$, $P_{10}$ dropped and row 1 renormalised): $\|O_{\rm full}-O_{\rm mask}\|_\infty=0.5263157894736843$ against $\varepsilon/(1-\gamma)=1.000$, with the naive $\varepsilon=0.1$ violated by $5.26\times$; the rule that every F1 union bound carries $1/(1-\gamma)$; the units rule ($\delta\,\|V\|_\infty$ with $\|V\|_\infty$ printed — the $\approx7.6\times10^{-5}$ of `sec_cost.md` §4.x.3 carried $\|V\|_\infty\approx5$ `[ASSUMED]`); the census guard $\|\hat P\|_\infty>1$ implies no certificate; and the sentence that the exact solve on an F1 mask is refused for fill-in.
*prove* DERIVED; `mask_amplification` `[S]` is J-L12's.
*measure* the $s=3$ instance; a random $s=32$ draw did **not** violate the naive bound (`RUN[J]`), which is why the plant is the closed form and never a random draw (V-10).
*PASS* written with the instance. *KILL* none this round; the row stays dormant until a mask ships (D-4).
*price* $0$ GPU-s · *mechanism* V-10, V-17, L-CERT, V-25 · *deliverable* `docs/CEQ_SHAPE.md` §4.2 Prop. 4(ii) · *evenings* 1

**S-01 ★ · Extend the identity manifest so a cell cannot be journalled without its boundary condition** — phase 0 · prereq none · independent-of every card · parallel-safe · aliases M-0.2, R-02, N-04
*build* the field list of `judge/sec_apparatus.md` §A.10 for the shape lane, extending `CONFIG_FIELDS` (`READ scale/identity_manifest.py:67-71`, which carries `beta` but not `qk`, `g`, `gamma`): `beta, qk, g`; `gamma` (init, final, $\Lambda$, verdict); `diag_convention` in `{kept, removed}`; `committor_route` in `{dirichlet_gamma1, discounted_gamma}`; `route` in `{solve_triangular, neumann_K, segmented, csr}` with $K$, chunk $C$, tile $B$; `boundary_sets, goal_set, sink_set` as sorted lists with a sha256 each; `K, m, t_star`; `S, D, d_model, n_train, n_eval, steps`; `seed, rng_plan`; `device, threads, dtype, torch_version, cublas_workspace, deterministic_regime`; `instrument_hash, manifest_hash`; `producer_cmd`; `delta_vec, delta_bare, one_over_1mg, V_inf`; the `census` block; `floor_exact, floor_zeroinfo, floor_fano_k1, ceiling_hop_k`; `sky_depth, sky_width, sky_cot, chacal_gamma, chacal_diag, params_per_arm`; `void_contrasts`; `supersedes`. Invariant: a missing declared field is a **refusal**, not an `absent` entry — the record's reason for reporting (stored manifests would be invalidated, `READ :146-148`) does not apply to a lane with no stored manifests.
*prove* nothing; an instrument.
*measure* the drift plant: flip each of `beta / qk / g / gamma / boundary_sets` by one unit and read `manifest_hash`; the pattern is `test_moving_one_citation_by_one_line_fires_both_binds` (`READ V20_R15_WING_MANIFEST.md:148-152`).
*PASS* `5/5` flips move the hash and an omitted `sink_set` raises. *KILL* any flip leaving the hash unchanged, or an omission returning `absent`: the lane has no identity, no cell may be journalled, and every certificate printed under that field is unattributable (L-2's recurrence: `0 of 24` attributable, `READ V20_R15_LEAP_LEDGER.md:23`).
*price* $0$ GPU-s · *mechanism* L-2, P-1, V-16, M-10, M-16 · *deliverable* `docs/apparatus/MANIFEST_SHAPE_LANE.md` · *evenings* 1

**S-02 · The identity-script rule: raise on a missing key, never default** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* a written rule for every identity check in the lane: bed keys are read by name and a missing key raises; no silent default. The filed instance: a scratch identity script asked `bed_1.build` for `basin_A`/`basin_B`, which it does not return (its keys are `A`, `B`, `READ ceq/beds/bed_1.py:167`), defaulted silently, and printed $0.858$ for an identity that reads $0.0$ on the real sets.
*prove* nothing.
*measure* must-fire: a misspelt key raises. Must-not-fire: the real keys return $\max|q-(I-Q)^{-1}R\mathbf 1|=0.0$ (`RUN[coord]`).
*PASS* an exception on the misspelt key and $0.0$ on the real one. *KILL* a number returned on a misspelt key: every identity number in the lane is suspect until the rule is enforced.
*price* $0$ GPU-s · *mechanism* V-16, V-3 · *deliverable* `docs/apparatus/IDENTITY_SCRIPT_RULES.md` · *evenings* 0.5

**S-03 · Print the obstruction vacuity inequalities beside every obstruction citation** — phase 0 · prereq none · independent-of every card · parallel-safe · aliases M-0.3 (obstruction half), R-03
*build* a one-page census of the three unconditional bounds as predicates on the record's geometry ($s=64$, $d=16$, $h=1$, $p=32$): $h\,m\,p=512\ge64$ (`sanford-2024-inductionheads` Thm 1), $n\log_2 n=384<544=H(d+1)p$ (`peng-2024-transformer-limitations` Thm 1), $64^{1/16}=1.2968$ against $Hdp=512$ (`chen-2024-multilayer` Thm 1.1) — all vacuous here; the conditional bound (`sanford-2024-logdepth` Cor. 4.3, the one-versus-two-cycle conjecture) named conditional in the same sentence.
*prove* DERIVED arithmetic.
*measure* the fraction of drawn BED-S cells whose geometry satisfies each hypothesis.
*PASS* every obstruction sentence carries its line and its `[V]`/`[U]` mark; $0$ per cent admitted means the sentence is stated as asymptotic lineage only. *KILL* (K-10) an obstruction stated without its line is struck at assembly.
*price* $0$ GPU-s · *mechanism* V-25, P-10, P-3 · *deliverable* `docs/apparatus/OBSTRUCTION_CENSUS.md` · *evenings* 0.5

**M-0.3 · Freeze the price ledger** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* `docs/PRICE_LEDGER.md` carrying §5.7's basis with provenance: the two fitted throughput laws with their R-squared and their fitting range, the solve increment as a per-op floor with the $2.0\times$ to $6.6\times$ dispatch gap beside it, the unit prices, the residency model with its `[ASSUMED]` operator-constant band, and the rule that no price crosses the device boundary (V-22).
*prove* nothing.
*measure* nothing; the ledger is read, not computed.
*PASS* every price carries a tag (`[MEASURED]`, `[FITTED]` with R-squared, `[RUN]`, `[ASSUMED]` with its reason, `DERIVED` with its arithmetic) and a producer. *KILL* any price without a tag or a producer is struck; a price quoted at $s>64$ before S-66 runs is struck (K-9).
*price* $0$ GPU-s · *mechanism* P-1, P-8, M-8, V-22 · *deliverable* `docs/PRICE_LEDGER.md` · *evenings* 1

**N-03 · The kernel-path must-fire battery, as a test specification** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* one test-module specification, CPU float64, each check with its honest half, its planted negative and the $O(1)$ failure the plant must produce: (1) parity `torch.equal(O(0), PV)` against the lane's own `softmaxAttn`, rejection $2.3002850040264393$ at $\gamma=0.5$; (2) `solve_triangular` against the dense inverse at $\le1.8\times10^{-15}$, plant a dense $M$ passed with `upper=False`; (3) the certificate in vector units with $\|V\|_\infty$ printed, plant rows scaled to $1.5$ at $\gamma=0.6$, $K=2$ reading $7.29$ against $0.54$ (convergent); (4) segmentation zeros by `torch.equal` on a masked column, plant a $-30$ logit that leaves the block non-zero; (5) CSR fill-in: a schedule dropping tile $(k,l)$ while keeping $(k,m),(m,l)$ makes the exact solve's $(k,l)$ block non-zero and the exact route is refused; (6) the three-outcome determinism table is read from N-01, never re-run here; (7) a missing key raises (S-02).
*prove* nothing; the theorems behind (1), (2), (4) are cited only once they build (L-LEAN).
*measure* the failure set, never the pass count (`sec_measured.md` M.9).
*PASS* every honest half passes and every plant fails at $O(1)$ with counts printed. *KILL* a plant that passes strikes that check from the battery and the paper's corresponding must-fire sentence.
*price* $0$ GPU-s · *mechanism* V-24, V-3, V-2, V-16, V-17 · *deliverable* `docs/CEQ_KERNEL_PATH_BATTERY.md` · *evenings* 1

**R-15 · Recount parameters per arm; fix the widths of the unmatched skylines** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* a per-arm parameter count printed in every table header (Ruling 3); the $4{,}769$ of `READ CEQ_V20_R15_CONTRACT.md:119` recounted, never assumed, once a $[m,K+2]$ head or a vector readout is attached; the wide constant-depth skyline's width fixed at $n_{\rm nodes}$ (`yehudai-2025-depthwidth` `[V]`) with the sentence that at $d_{\rm model}=16<s=64$ the matched-parameter instance does not exist; the chain-of-thought decoder's step count fixed at $t^\star$ (`merrill-2024-cot` `[V]`).
*prove* nothing.
*measure* the count per arm and the residual mismatch.
*PASS* every matched arm within $0.032$ per cent of the reference count (`READ MODEL_CARD.md:76-79`). *KILL* a "matched" contrast above that residual is moved to the skyline column and never credited; an arm re-architected to close the gap is refused (Ruling 3).
*price* $0$ GPU-s · *mechanism* M-8, `MISTAKES.md` D-1, R-SKY · *deliverable* `docs/plan/PARAM_TABLE.md` · *evenings* 1

**R-17 · Freeze every guard partition before any itinerary is read** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* the $q=\tfrac12$ isocommittor guards (`READ ceq/beds/bed_1.py:5-8, :382`) and any BED-S basin partition fixed from the oracle's committor **before** any itinerary is scored, with the partition's sha256 in the manifest; the Pesin-deficit reading kept as the diagnostic it is (the record's contract guards had deficit $0.009654$, $11.9$ per cent of $h$; the state partition $0.000233$, `READ V15_BED1.md:248-272`) and never used to select guards by argmin.
*prove* nothing.
*measure* the deficit of the frozen partition; the guard-crossing balance (the record's $21$ net on $65{,}481$ events).
*PASS* the partition hash precedes the first itinerary record in the journal's order. *KILL* a partition whose hash postdates an itinerary it scores: every itinerary statistic on it is struck as a threshold refitted to the data it judges (M-2).
*price* $0$ GPU-s · *mechanism* M-2, M-18, M-19 · *deliverable* `docs/plan/GUARD_FREEZE.md` · *evenings* 1

**S-61 ★ · The $\gamma$-pinning likelihood-ratio instrument, with the boundary null** — phase 0 · prereq none · independent-of every card · parallel-safe · aliases J-D9, M-4.3 (spec half), R-11, V-8, ledger J-1
*build* the specification: $\Lambda=2[\mathrm{LL}(\hat\gamma)-\mathrm{LL}(\gamma\equiv0)]$ on a declared held-out set of $n_{\rm eval}=4096$ (Ruling 10-prime restated for $\gamma$, which was stated for $\beta$); with $\gamma$ trained on $[0,1)$ the null sits on the parameter boundary, so $\Lambda$ is asymptotically $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ whose 95 per cent point is the $\chi^2_1$ $0.90$ quantile $2.705543$, **not** $3.841459$; PINNED at $\Lambda\le2.7055$, MOVED at $\Lambda>\ln n_{\rm eval}=8.318$, the interval verdict between; the seed rule "MOVED on at least 6 of 8" and the SPLIT band $3/8$ to $5/8$ declared here; the ablation $(I-\hat\gamma\hat P)^{-1}\to I$ at trained weights must move NRMSE by more than one seed sd; the mirror kill $\hat\gamma>0.99$ on at least 6 of 8 prints $1/(1-\hat\gamma)$ beside every $\delta$; the $|\hat\gamma|<0.05$ rule is **deleted** (two verdicts for one cell, M-20). The citation for the boundary null is **owed** — not in `references.bib`, not fetched — and the paper may not cite it by name until it is.
*prove* nothing; the quantile is `RUN[J]`, `RUN[MARS]`, `RUN[MERCURY]` to the printed digits.
*measure* the instrument's own plants on synthetic data: a generator at $\gamma_{\rm env}=0$ must read PINNED on $8/8$ seeds and one at $\gamma_{\rm env}=0.6$ must read MOVED on $8/8$.
*PASS* both plant counts $8/8$. *KILL* either fails: Bet C cannot be scored and is VOID until the instrument is repaired; a threshold chosen after $\Lambda$ is seen is M-2 and strikes the verdict.
*price* $0$ GPU-s for the spec, seconds for the plants · *mechanism* Ruling 10-prime, V-9, M-2, M-20, V-17, V-24 · *deliverable* `docs/apparatus/INSTRUMENT_GAMMA_LR.md` · *evenings* 1

**V-0 · Open the calibration ledger and carry the R11 column into it** — phase 0 · prereq none · independent-of every card · parallel-safe
*build* a ledger with one row per bet: bet, source in `{author, leap}`, prediction, counter, deciding number, frozen PASS, frozen KILL, realised, verdict in `{HOLDS, COUNTER, SPLIT, VOID}`, sign of miss, date, producer command; the R11 nine rows carried in verbatim as the first block (`READ V16_CALIBRATION.md:40-48`), each with its sign; the five D-CALIB rules printed at its head.
*prove* nothing; an instrument.
*measure* the running one-sided sign test after every scored row; the column starts at $7/8$ optimistic, $p=0.0352$, Wilson 95 per cent $[0.5291, 0.9776]$.
*PASS* the ledger exists with the nine rows and the five rules. *KILL* a row filed without its counter is refused at the file, never discounted (D-CALIB-2); what dies is the row, not the bet.
*price* $0$ GPU-s · *mechanism* D-7, M-2, P-1 · *deliverable* `docs/CALIBRATION.md` · *evenings* 1

---

### PHASE 1 — BED-S: specification, census, admission

**Goal.** Build the one bed on which the shape's three capabilities have a rejection region, and decide from a printed census whether it is admissible — before a GPU-second is spent. The record's dominant failure is a bed that dies at construction (D-4, M-3, V-8); this phase makes that death free.

**The gate (D-4 admission).** The author reads one census block and admits or reroutes. Admit requires, on 512 draws at the design point $t^\star=8$, $m=8$, $K=2$, $s=64$: row sums within $10^{-12}$ of one including absorbing rows; $0\in\mathcal{A}_{\rm sink}$ on $100$ per cent; $\rho(\hat Q)=\max_{i\in T}\hat P_{ii}<1$ on $100$ per cent; every boundary set before the query and the query in $T$ on $100$ per cent; every per-coordinate label sd over the **admitted query region** above $0.05$; every argmin class frequency in $(0.05,0.95)$; rule-disagreement fraction above zero; the sink share printed apart from the goal share; the discard count printed. Any $\mathrm{sd}=0$, any class outside the band, disagreement zero, or $t^\star$ unplaceable inside the causal window kills admission (K-E2, K-P): no BED-S reading exists, the reroute is the jittered `bed_1` landscape at $K=2$ with $B$ as goal used as an **oracle cross-check only**, and $N$ is repriced from whatever realised sd the first admitted batch shows. Price of the whole phase: $0$ GPU-s, except S-14's single zero-step forward.

**S-10 · Register the placement convention: BOS is a value-zero sink set, boundary sets precede the query** — phase 1 · prereq none · independent-of S-01 to S-03, J-*, M-0.3 · parallel-safe
*build* the two construction facts as a written convention. **F1**: row 0 of any causal softmax at $\beta=1$ is $e_0$, so position 0 is absorbing whether or not it is declared; undeclared, the transient block is singular. **F2**: walks descend, so a boundary set after the query is unreachable. BED-S therefore declares $\mathcal{A}_{\rm sink}=\{0\}$ with value $0$ on every indicator channel — **not** a goal member, since with $0\in\mathcal{A}_0$ "reach the goal" collapses to "descend to 0 without hitting a constraint" — and places the sink, $\mathcal{A}_0$ and $\mathcal{A}_1..\mathcal{A}_K$ all before the query, the query in $T$.
*prove* both facts are theorem instances once J-L2 and J-L4 build; until then they are RUN.
*measure* with BOS undeclared, $\rho(Q)=1.000000$ and $\det(I-Q)=0.0$ — the solve must raise; with $\mathcal{A}_{\rm sink}=\{0\}$ declared, $\rho(Q)=0.692660$, the conservation row holds to $4.4\times10^{-16}$ and the sink share reads $[0.362, 1.000]$ on that draw; a set at position 20 read from query 12 gives $q=0.0$ exactly (discard), from query 25 gives $q=0.1033$ (`RUN[SATURN]`).
*PASS* the four readings reproduce on the bed's own generator at $s=64$. *KILL* a declared-sink draw with $\rho(Q)=1$: the generator has a second undeclared absorbing row (a zero gate at $c$, J-L7) and census line 14 must be re-read before admission.
*price* $0$ GPU-s · *mechanism* V-25, V-12, V-8, D-3 · *deliverable* `docs/beds/BED_S_PLACEMENT.md` · *evenings* 0.5

**S-11 · Write the BED-S generator specification** — phase 1 · prereq S-10 · independent-of S-01 to S-03, S-50 to S-53, every J- card · parallel-safe · aliases M-1.1, R-04 (spec half), V-3 (spec half), ledger S-1
*build* the environment is a random directed graph on the $s$ token positions whose edges point to
**earlier** positions — a DAG in token order, so a walk from the query descends as the arm's causal $\hat P$ does — with self-loops only on declared absorbing positions and $P_{\rm env}$ row-stochastic including them. Node tokens carry the out-adjacency as a multi-hot feature, the membership flags of the absorbing sets, and **nothing derived from any solve**; the transition matrix is a deterministic function of the edge tokens, so leak clause (a) is dropped for BED-S and clause (c) is kept as the kill. Candidate moves are $m$ **query-side row clamps** $P_{\rm env}[v_a,:]\leftarrow e_{u_a}$, each rank one with $u^\top\mathbf 1=0$ (the graph-surgery precedent, `READ scale/negation_scope.py:718-732`); a token rewrite is rank $s-i$, a different object priced as a suffix re-solve. Label per move: the reach-avoid tensor $(q^{({\rm sink})},q^{(0)},q^{(1..K)})$ at $\gamma=1$ on the transient block, the full vector on $T$ retained for the residual; $a^\star=\arg\max_a q^{(0)}$, the Chebyshev $\arg\min_a\max_{k\ge1}q^{(k)}$ and the lexicographic form as **columns**; $z^\star$ and $\Delta z(a)$ by the displacement identity on $P_{\rm env}$ at a bed constant $\gamma_{\rm env}$. Dials with registered supports: $t^\star\in\{2,8,32\}$ (a **tolerance** dial read off the hop ladder, never predicted from a spectrum), $K\in\{2,3,4\}$ ($K=1$ is illegal: the committor is constant to $1.11\times10^{-14}$), $m\in\{4,8,16\}$ default $8$, $|\mathcal{A}_\bullet|\in\{1,2\}$, $s\in\{64,256,1024,4096\}$. Oracle: the Dirichlet solve of `bed_1.committor`'s form (`READ ceq/beds/bed_1.py:188-198`) on $P_{\rm env}$, never on $\hat P$; the Kirchhoff second route covers $K=2$ single-node sets only, the multi-node extension is `NOT MEASURED, needs the grounded Laplacian extended`.
*prove* nothing here; the identities are Phase 2's.
*measure* nothing yet; the spec names every field S-01's manifest consumes and every census line S-12 prints.
*PASS* the spec states the oracle runs on $P_{\rm env}$ and names the D-4 registration triple on the same commit. *KILL* any sentence handing the arm $P_{\rm env}$ as an input channel other than the edge tokens (D-2), or a spec in which $K=1$ is legal (V-12).
*price* $0$ GPU-s · *mechanism* D-2, V-25, V-12, V-8, D-3, M-8 · *deliverable* `docs/beds/BED_S_SPEC.md` · *evenings* 2

**S-12 ★ · Run the BED-S domain census on 512 draws and decide admission** — phase 1 · prereq S-11 · independent-of S-01 to S-03, S-50 to S-53, every J- card · parallel-safe · aliases M-1.2, R-04, V-3, ledger S-2, P1.2
*build* the fifteen census lines of `judge/sec_apparatus.md` §A.7 printed per draw batch as one JSON block, before any arm is trained: (1) row stochasticity including absorbing rows; (2) $0\in\mathcal{A}_{\rm sink}$ on $100$ per cent; (3) $\rho(\hat Q)<1$ as a diagonal read — a V-10 gate, printed, never counted as evidence; (4) every set before the query, query in $T$; (5) per-coordinate label sd over the **admitted query region** (the prefix before the first constraint reads $q^{(0)}\approx1$ and would inflate it), sink share printed apart from $q^{(0)}$; (6) every argmin class frequency in $(0.05,0.95)$, ties within $10^{-9}$ discarded and counted; (7) rule-disagreement fraction above zero; (8) the corpus-alone probe of S-15; (9) $I(s_0;a^\star)=0$ by plug-in; (10) $\mathrm{Var}(\Delta z)>0$ over coordinates $\ge i_{\min}$ with the zero-coordinate fraction printed; (11) the move census (which moves change $P$ against $V$ only; $u^\top\mathbf 1=0$ and a non-negative edited row on $100$ per cent; the Sherman-Morrison denominator printed as a check); (12) and (13) filled on trained cells; (14) the exact-zero-gate count (expected `0 of N` on the softmax corner, so the F0 theorems are silent); (15) S-03's lines. `nrmse` returns `nan` on a constant label and `nan >= 1.0` is `False` (`READ MISTAKES.md:149-153`), so the sd print is not optional.
*prove* nothing; F1 and F2 appear as the two structural lines.
*measure* 512 draws at $t^\star=8$, $m=8$, $K=2$, $s=64$.
*PASS* every admitted-region sd above $0.05$; every class frequency in $(0.05,0.95)$; disagreement above zero; lines 1 to 4 at $100$ per cent; the discard fraction printed. *KILL* (K-E2, K-P) any $\mathrm{sd}=0$; any class outside the band; disagreement zero (then only one rule may be named); $\rho(Q)=1$ on any draw (BOS undeclared: the solve must raise, V-16); $t^\star$ unplaceable. The bed is not admitted, **no arm runs**, every card from S-14 to S-65 is VOID, and §5.10 tree C is the paper.
*price* $0$ GPU-s; the oracle is $m$ factorisations with $K+2$ right-hand sides, $|T|^3/3=5.76\times10^{8}$ MACs at $|T|=1200$, under a second per draw in float64 (DERIVED) · *mechanism* V-8, V-12, V-25, V-10, D-3, D-4, M-3 · *deliverable* `results/bed_s_census.jsonl` and `docs/beds/BED_S_CENSUS.md` (the admission decision, signed by the author) · *evenings* 1

**S-13 · Register the VOID-contrast list and the identification metric before any cell** — phase 1 · prereq S-11 · independent-of S-12 · parallel-safe · aliases M-1.3, R-16, ledger S-3
*build* because the arm's operator class contains the oracle's chain, `shape - softmax` and `shape - skyline` on the committor head are reproduction-versus-non-reproduction contrasts and are
**VOID as capability numbers**. Creditable: `shape - ChaCAL-diag`, `shape - ChaCAL-published`, `shape - ChaCAL-with-sink-token`, `shape - InfSA-Neumann-16`, `shape - 0-hop MLP` and `shape - 1-hop softmax` on the argmin, and $\|\hat P-P_{\rm env}\|_\infty$ per seed as a
**learnability** reading. The model is E1's registration comment (`READ scale/negation_scope.py:1033-1041`).
*prove* J-L9 is the D-2 separation that fails on an undirected substrate and holds on the DAG.
*measure* (at the arena) $\|\hat P-P_{\rm env}\|_\infty$ per seed, sorted.
*PASS* the list is in the manifest field `void_contrasts` on every cell, in a commit dated before the first arena cell. *KILL* (K-D2) $\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at least 6 of 8 seeds: the arm copied the environment, every contrast against a non-reproducing arm is a copy-versus-no-copy statement, and the reading is filed as identification, never as capability.
*price* $0$ GPU-s · *mechanism* D-2, M-7, V-10 · *deliverable* `docs/beds/BED_S_VOID_LIST.md` · *evenings* 0.5

**S-14 · The zero-hop guard: the query carries $s_0$ only, and the untrained arm reads chance** — phase 1 · prereq S-11 · independent-of S-12, S-13 · parallel-safe · aliases M-1.4, ledger P1.3, X-9
*build* $s_0$ drawn uniformly from $T$ independently of $a^\star$; the plug-in $I(s_0;a^\star)$ printed with its sampling error; the zero-step RED gate applied unchanged — the untrained arm within $1/m$ plus or minus its Clopper-Pearson half-width on the argmin, committor NRMSE $\ge1-\mathrm{GATE\_TOL}$ with $\mathrm{GATE\_TOL}=10^{-3}$ (`READ COSTS.md:137`). Precedent: BED-M's first builder leaked $1/(t^\star+1)$ at zero hops and aborted three of five rungs; $b[s-1]=0$ restored it (`READ scale/negation_scope.py:399-415`).
*prove* nothing.
*measure* must-fire: a query token carrying $q^{(0)}(a^\star)$ pushes the untrained read below $1-\mathrm{GATE\_TOL}$. Must-not-fire: the honest query reads at chance within the interval.
*PASS* both halves. *KILL* the honest query reads below the gate: the label is legible at zero hops, the bed is a copy task, and it is struck before any number.
*price* $\le1.524$ s, one zero-step forward `[FITTED]` · *mechanism* V-10, D-5, V-24 · *deliverable* `results/bed_s_zerohop.jsonl` · *evenings* 0.5

**S-15 · The leak guard, firing both ways** — phase 1 · prereq S-11 · independent-of S-12 to S-14 · parallel-safe · aliases M-1.5, R-05, ledger P1.4, X-8
*build* a corpus-alone linear probe from the token features to $q$ at order 0 on the admitted draws; the planted positive is a move token carrying its own $q$; the planted negative is the strictly-local feature set E4-prime uses (degree, ball sizes to radius 5, absorbing-endpoint-in-ball flags, `READ scale/e4_harmonic.py:220-244`), whose bars are E4-prime's and are **re-derived for BED-S** before use (V-22).
*prove* nothing.
*measure* the coefficient of determination on the honest probe and on the planted leak.
*PASS* honest below $0.5$ and planted at or above $0.99$ — the detector fires when the leak is planted, calibrated both ways. *KILL* honest at or above $0.5$: the one-read claim is void on this bed (leak kill (c)); planted below $0.99$: the detector is blind and no leak verdict is admissible.
*price* $0$ GPU-s (CPU probes) · *mechanism* M-21, V-24, V-7, V-22, D-2 · *deliverable* `results/bed_s_leak.jsonl` · *evenings* 1

**S-16 · Fix the metrics and the floors before any arm exists** — phase 1 · prereq S-11 · independent-of S-12 to S-15 · parallel-safe
*build* the committor head scored in the Fisher-Rao coordinate $\varphi(p)=2\arcsin\sqrt p$ per entry, position-matched NRMSE on $\varphi$ (mean and max), plus the harmonic residual $r(\hat q)=\|(I-Q_{\rm env})\hat q-R_{\rm env}\mathbf 1\|_\infty$ — a **score, never a loss** (D-2) — printed beside $\sigma_{\min}(I-\gamma P_{\rm env})$ and $\|I-\gamma P_{\rm env}\|_\infty\le1+\gamma$, its units. Argmin accuracy with Clopper-Pearson (`clopper-1934-binomial`) and McNemar paired on identical draws (`mcnemar-1947-correlated`). The $z$ and $\Delta z$ channels by the position-matched per-coordinate NRMSE vector, the field cosine and the magnitude ratio over coordinates $\ge i_{\min}$ with the masked fraction printed (an arm emitting $\Delta\hat z\equiv0$ makes the cosine undefined and is **refused**, not scored, V-16), the sign column kept because softmax owns it ($0.807843$, `READ MATHEMATICS.md:396-406`), McNemar on the sign column only and a paired $t$ or Wilcoxon on the cosine with its **own** realised sd. Floors (L-FLOOR): the exact oracle at $0.0$ with the hop-ladder ceilings printed per batch; the argmin zero-information floor $1-\max_a\hat\pi(a^\star)$, which at uniform $m=8$ is $0.875$; the tight Fano $(\ln m-I-\ln2)/\ln(m-1)$, which at $m=8$ with $I=0$ reads $0.7124$, used **only** where $I(X_{\le k};a^\star)>0$ is computed; the weak Fano $0.6667$ retired as a floor (it sits $0.208$ below chance); `floor_1` nowhere. Refused metrics: pooled first-order Wasserstein (the permuted oracle scores $0.0$ against NRMSE $1.421901$), Procrustes, position-free optimal transport.
*prove* nothing.
*measure* nothing; the file is hashed into the prediction file of S-60 before the first arena cell.
*PASS* the metric file exists and is hashed before S-62. *KILL* any metric added or removed after the first arena cell (M-2); that cell's numbers are then not quotable.
*price* $0$ GPU-s · *mechanism* M-2, V-26, V-17, L-FLOOR, V-16 · *deliverable* `docs/apparatus/METRICS_AND_FLOORS.md` · *evenings* 1

**S-17 · The $\Delta z$ channel's own census and the cached-mixture plant classes** — phase 1 · prereq S-11 · independent-of S-12 to S-16 · parallel-safe
*build* for the $z$ and $\Delta z$ lane — its **own** journal, because a vector label voids the record's published softmax baseline (`READ MISTAKES.md:701-708`) — the intervened position drawn uniformly; $\mathrm{Var}(\Delta z)>0$ per coordinate $\ge i_{\min}$; per-draw flags `move_changes_P` against `move_changes_V_only` so the cached-mixture control can be scored on the two plant classes separately; the no-change predictor's error printed beside every $\Delta z$ number (`vakalis-2026-interventiongap` `[V]` owns that floor).
*prove* nothing.
*measure* the two plant classes' counts.
*PASS* both classes non-empty, at least 64 draws each. *KILL* a corpus with no $P$-changing move: the interventional channel has no rejection region against the cached mixture and card S-33 cannot be scored.
*price* $0$ GPU-s · *mechanism* V-8, D-5, V-24, M-1 · *deliverable* `results/bed_s_dz_census.jsonl` · *evenings* 0.5

**R-06 · Register the zero-hop, one-hop and hand-rule controls at construction** — phase 1 · prereq S-11 · independent-of S-13 to S-17 · parallel-safe
*build* the argmin controls (0-hop per-position MLP on the candidate's context row, 1-hop softmax, majority, random at $1/m$, predict-the-mean) plus R-24's closed-form rule, all registered before any arm trains; the exact zero-information floor from the realised class distribution; $I(X_{\le0};a^\star)$ by plug-in over the zero-hop view **including** the move tokens and membership flags.
*prove* nothing.
*measure* the untrained arm at $1/m$ within its interval; committor NRMSE $\ge1-\mathrm{GATE\_TOL}$ at step 0.
*PASS* the untrained arm reads chance and $I(X_{\le0};a^\star)$ is printed, with the tight Fano used only where it exceeds zero. *KILL* (K-H2, scored at S-26) the 0-hop MLP within the Clopper-Pearson half-width of the shape at $N=8$, or McNemar $p>0.05$ against the 1-hop softmax: BED-S is a third static task and is struck before any number is quoted (C8; the record's static beds gave an iterating arm nothing to compute toward, `READ D1.md:286-291`).
*price* $0$ GPU-s at registration; $\le1.524$ s per control cell when trained `[FITTED]` · *mechanism* V-10, D-5, C8 · *deliverable* `docs/plan/CONTROLS_REGISTERED.md` · *evenings* 1

**R-07 ★ · Census the sink-escape degeneracy of the Chebyshev rule and the exact-zero tie it creates** — phase 1 · prereq S-11 · independent-of S-13 to S-17, R-06 · parallel-safe
*build* the attack, filed by no office before: with $\mathcal{A}_{\rm sink}=\{0\}$ declared and moves realised as clamps, a clamp into **any** absorbing position gives $\max_{k\ge1}q^{(k)}=0$ exactly, so the Chebyshev column is minimised equally by every move that sends the walk to the sink or to the goal — the safest move under Chebyshev is "fall off the prompt" — and the $10^{-9}$ tie rule then discards the draw, which empties the bed. Two census lines are added to S-12: (i) the fraction of admitted draws on which $a^\dagger$ has $q^{({\rm sink})}(a^\dagger)>q^{(0)}(a^\dagger)$; (ii) the discard fraction attributable to exact-zero Chebyshev ties. The construction fix is registered
**now**, before the number: clamp targets restricted to transient positions ($u_a\in T$) so no move is a direct absorption, and a third column, the conditioned Chebyshev $\arg\min_a\max_{k\ge1}q^{(k)}(a)/(1-q^{({\rm sink})}(a))$, printed beside the other two.
*prove* DERIVED: for $u\in\mathcal{A}_j$ the clamp gives $q^{(j)}(v)=1$ and $q^{(k)}(v)=0$ for $k\ne j$ exactly, so the Chebyshev value of every absorbing-target move is zero unless $j\ge1$.
*measure* lines (i) and (ii) on 512 draws, with and without the restriction. On the judge's draw at query 20, over the 20 clamp targets below it, the Chebyshev minimum is $0.000$, attained by `9 of 20` targets, of which 7 carry sink share above $0.5$ and 2 goal share above $0.5$ (`RUN[M]`).
*PASS* with the restriction, line (i) below $0.05$ and line (ii) below $0.05$. *KILL* line (i) at or above $0.5$ on the unrestricted bed: the Chebyshev column as written is struck and only the goal rule and the conditioned form are named; line (ii) at or above $0.5$: the discard rule slices the bed to nothing and admission is refused until the restriction is in the spec.
*price* $0$ GPU-s · *mechanism* V-12 in a new guise, V-8, V-10, V-5 · *deliverable* `docs/plan/SINK_ESCAPE.md` and two columns in `results/bed_s_census.jsonl` · *evenings* 1

**R-08 · Count no-op moves and correct the chance floor to the effective $m$** — phase 1 · prereq S-11 · independent-of R-05 to R-07 · parallel-safe
*build* the attack: on a causal $P_{\rm env}$ a clamp at a row above the query cannot change any committor at the query, and a clamp at a row unreachable from $s_0$ is a no-op for the same reason — F2 applied to **moves**, which no office filed (F2 covers boundary sets). A bed with $m=8$ of which $k$ are no-ops has effective $m-k$ candidates, so the printed chance floor $1-1/m=0.875$ and the tight Fano $0.7124$ are too **low**, and a shape reading above the printed floor may be at chance on the effective set. The census line: per move, $\max_k|q^{(k)}(\mathrm{do}\,a)-q^{(k)}|$ at the query, a move below $10^{-12}$ counted a no-op; $m_{\rm eff}$ per draw; both floors recomputed at $m_{\rm eff}$; the spec restricted to clamp rows at or below $s_0$ and reachable from it, with the fraction of draws touched by the restriction printed.
*prove* DERIVED from triangularity, and by J-L4's `later_boundary_unreachable` extended to a clamp row.
*measure* $m_{\rm eff}$ on 512 draws; clamps at rows 21, 25, 31 move the query's four committors by $5.551\times10^{-17}$, zero to rounding (`RUN[M]`).
*PASS* $m_{\rm eff}=m$ on at least 95 per cent of admitted draws after the restriction. *KILL* $m_{\rm eff}<m$ on more than 5 per cent of draws in any cell quoting a floor: every argmin accuracy in that cell is re-floored at $m_{\rm eff}$ and this census line becomes an admission line.
*price* $0$ GPU-s · *mechanism* L-FLOOR, V-10, V-17, D-3 · *deliverable* `docs/plan/EFFECTIVE_M.md` and one column in `results/bed_s_census.jsonl` · *evenings* 1

**R-24 · Register the clamp-target membership rule as a closed-form zero-hop control** — phase 1 · prereq S-11 · independent-of R-05 to R-08; consumed by R-06 and S-62 · parallel-safe
*build* the attack: the goal rule is decided at zero hops whenever some clamp target lies in the goal set, because a clamp into $\mathcal{A}_0$ gives $q^{(0)}=1$ exactly and the goal flag of $u_a$ is a token feature. The registered 0-hop control is a **trained** MLP; a closed-form rule R0 — "pick the move whose clamp target carries the goal flag; else the move whose target carries no constraint flag and sits earliest" — costs nothing, needs no seed, and reads its accuracy at construction. R0 and its two ablations (goal-flag only; earliest-transient only) are computed as oracle-free columns on every draw batch, with the fraction of draws on which R0 is defined.
*prove* DERIVED: a clamp into $\mathcal{A}_0$ yields $q^{(0)}=1$ and is the unique argmax unless another move also clamps into $\mathcal{A}_0$; on the judge's draw $q^{(0)}(v)=1.0000$ exactly for $u=5\in\mathcal{A}_0$ (`RUN[M]`).
*measure* R0 accuracy against the oracle argmin with Clopper-Pearson on 512 draws, before and after R-07's restriction.
*PASS* R0 accuracy within the interval of $1/m$ after the restriction. *KILL* (frozen) R0 accuracy at or above $\mathrm{acc}_{\rm shape}-\mathrm{MDE}_8$ on at least 6 of 8 arena seeds, or R0 above the exact zero-information floor by more than its interval on the unrestricted bed: the argmin label is a flag lookup, BED-S's argmin head is struck, and only the committor-vector head remains.
*price* $0$ GPU-s · *mechanism* V-10, D-5, M-21, M-18 · *deliverable* `docs/plan/FLAG_RULE.md` and three columns in `results/bed_s_census.jsonl` · *evenings* 1

**R-20 · File the one-step rule as a safety filter, not a policy, with the multichain hazard tested on the oracle** — phase 1 · prereq S-11 · independent-of R-05 to R-08, R-24 · parallel-safe
*build* on 512 admitted draws, the oracle-only comparison of the one-step rule against the two-step optimum (apply $a$, re-solve, take the best second clamp) on the same $P_{\rm env}$; the vocabulary fixed as a one-step, most-restrictive **safety filter** whose value is the goal committor (`hsu-2023-safetyfilter`, `borquez-2023-lrf` `[V]`), the aggregations as columns (`vanmoffaert-2013-chebyshev`, `yang-2026-lexisafe` `[V]`), and `misra-2023-safety-constrained-mdp` `[V]` (Bellman's principle can fail for safety-constrained multichain MDPs) carried in Limits beside the rule.
*prove* DERIVED: on a DAG with declared absorbing sets every transient state is absorbed with probability one ($\rho(Q)=\max_T P_{ii}<1$), so the chain is absorbing; the hazard lives in the policy-level object, which the paper does not claim.
*measure* the fraction of draws on which the one-step and two-step optima disagree.
*PASS* the paper never writes "optimal move" or "policy", and the disagreement fraction is printed. *KILL* disagreement above $0.5$ on 512 draws: the phrase "safest move" is replaced by "one-step safety filter" in every sentence and the sequential problem is filed as future work with the citation.
*price* $0$ GPU-s (oracle solves) · *mechanism* P-7, V-17, P-10 · *deliverable* `docs/plan/SAFETY_FILTER.md`, `results/bed_s_onestep_vs_twostep.jsonl` · *evenings* 1

---

### PHASE 2 — the binds through the front door, the smoke test, the device certificate

**Goal.** Prove on the bed's **real draw** that every identity the paper leans on holds and every planted negative fails at $O(1)$, that the solve path runs deterministically at the predicted price, and that the device certificate has a live producer. Nothing here reads a capability number.

**The gate (V-14, V-24).** Every plant of S-20 to S-27 exists as a FOUND cell under `results/` with its own `kind` and `manifest_hash` — a plant that lives only in a script's print is NAMED, and the bind is inadmissible. Every honest half passes; every plant fails at $O(1)$ with counts printed, in the pattern $0.9749 / 0.9165 / 1.000 / 0.4845$ (`READ workdonenewseal.md:114-122`). The smoke cell reads inside the law-versus-measured band $[-1.8, +14.4]$ per cent of $1.680$ s, or is re-labelled the per-op floor it was declared to be. The author decides one thing: whether the arena may start.

**M-2.0 · Build the shape arm as a registered `kind`** — phase 2 · prereq none (building is not training; L-LEAN gates M-2.2, not this) · independent-of every Phase-1 card · not parallel-safe with S-20
*build* one module exposing the corners `beta/qk/g`, a learnable $\gamma\in[0,1)$ initialised at $0$, `boundary_sets/goal_set/sink_set` as identity rows, `diag_convention`, `route`, `committor_route`, the read $O=(1-\gamma)P(I-\gamma P)^{-1}V$ with the factor carried (`fagnou-2024-chacal` Eq. 5 `[V]`), and two heads: (H-q) the exact triangular solve at $\gamma=1$ on the transient block and (H-z) the state and displacement channel at $\hat\gamma$; plus the `make_arm` branch, the optimiser path and the journal emit — the W2 lesson was that a three-line branch was the whole gap (`READ V20_R15_WING_MANIFEST.md:115-123`).
*prove* nothing.
*measure* the arm produces a journalled record with its `kind` under `results/` (FOUND, not NAMED).
*PASS* one FOUND record. *KILL* a `kind` with zero records at the end of Phase 2 is STRUCK, which is the R15 kill applied to this lane.
*price* $0$ GPU-s to build; the first record is S-20's · *mechanism* FOUND-not-NAMED, P-4, L-2 · *deliverable* the author's module plus the FOUND line in `docs/PLAN_STATUS.md` · *evenings* 3

**S-20 ★ · B-J parity at $\gamma=0$ and the ChaCAL-diag smoke test at $\gamma=0.9$ on BED-M** — phase 2 · prereq S-01, S-02, M-2.0, J-L18 rows 1 and 5 (L-LEAN) · independent-of every Phase-1 card · not parallel-safe with M-2.0 · aliases M-2.2, R-12, V-7, N-05, ledger P2.1, B-J, B-P5
*build* `torch.equal(O(0), PV)` against the lane's own `softmaxAttn`; then one 150-step ChaCAL-diag cell at fixed $\gamma=0.9$ (ChaCAL's own setting, `fagnou-2024-chacal` App. C) on BED-M `e3_t2`, $n=2048$, $s=64$, $d_{\rm model}=16$, under `use_deterministic_algorithms(True)` with the cuBLAS workspace exported **before** process start (set in-process it does not take, `READ V17_R4_RETAKE_PRICE.md:178-181`), timed with `torch.cuda.synchronize()` bracketing. Plants: $\gamma=0.5$ (`RUN[coord]` $2.3002850040264393$ at $s=64$; `RUN[SATURN]` $1.6143$ at $s=32$); $\beta=0$ at the softmax corner (gap above $0.5$); ChaCAL-published's diagonal-removed inverse at any $\gamma>0$; and B-P5's dense non-triangular $M$ passed with `upper=False`. The struck plant — a non-causal $W$ — passes bitwise at $\gamma=0$ because $I-0\cdot W=I$ regardless of support (`RUN[M]`) and is recorded as struck.
*prove* nothing here; Proposition 1 is J-L1's.
*measure* `torch.equal` at $\gamma=0$; `solve_triangular` against the dense inverse $1.7763568394002505\times10^{-15}$; the four plant residuals; the cell wall-clock; bitwise repeats forward and backward over 8 repeats under strict mode.
*PASS* identity `True`; `4/4` plants at $O(1)$; the cell inside $[-1.8, +14.4]$ per cent of $1.680$ s; repeats bitwise both ways. *KILL* `torch.equal` `False` at $\gamma=0$ (the containment sentence dies); any plant passing (that bind is struck, V-24); the backward raising under strict mode (the "first gated wing whose training step runs strict" sentence is deleted and the lane runs `warn_only`); the cell above $2.2\times$ the law (every `[FITTED + RUN]` price in §5.7 is re-tagged the P-8 floor it was declared to be and re-measured before any GPU-minute figure is quoted). No capability number is read: BED-M is contained (J-L11, D-2).
*price* $\approx 17.4$ s for 8 seeds ($8\times1.681+4.0$), $\approx1.7$ s for a single cell `[FITTED + RUN]` · *mechanism* V-24, V-3, M-8, P-8, V-16, D-2 · *deliverable* `results/binds/b_j_parity.jsonl` with kinds `shape_g0`, `shape_g05_plant`, `shape_b0_plant`, `chacal_pub_plant`, `dense_M_plant` · *evenings* 1

**S-21 · B-E1 boundary rows against ChaCAL-diag** — phase 2 · prereq S-12 admitted, S-20 · independent-of S-22 to S-28 · parallel-safe · alias ledger P2.2
*build* identity half: with $\mathcal{A}=\emptyset$ the shape is `torch.equal` to **ChaCAL-diag**, the lane's own re-implementation with `diag_convention = kept` declared in the manifest (a declared V-3). Rejection half: with $\mathcal{A}\ne\emptyset$, $\|\Pi_{\rm shape}-\Pi_{\rm ChaCAL\text{-}diag}\|_\infty$ on rows downstream of the boundary positions exceeds a printed $O(1)$ gap — a **weight** statement, never a support statement, since on a dense causal softmax the support of $\Pi_\gamma$ equals the support of $P$ with or without boundary rows (`RUN[J]`, rows changed $=[3]$, `RUN[M]`). The struck plant — "a non-identity absorbing row breaks the conservation row" — is empty (`RUN[M]` $0.0$; the sum uses transient rows only) and is replaced by the full-$P$ read at $\gamma<1$, where it reads $0.1491$. The InfSA base must break the identity half: `NOT MEASURED, needs the base wired`.
*prove* nothing.
*measure* the identity half and the downstream gap on every admitted draw.
*PASS* `torch.equal True` at $\mathcal{A}=\emptyset$ and a downstream gap of at least $0.1$ at $\mathcal{A}\ne\emptyset$ on $100$ per cent of admitted draws. *KILL* the identity half fails (the lane's ChaCAL-diag is not the shape at $\mathcal{A}=\emptyset$ — a P-7 object, and the "same operator" sentence dies); the gap reads below $10^{-3}$ on any draw (boundary rows change nothing there; census line 4 is re-read).
*price* seconds · *mechanism* V-24, V-3, V-14, P-7 · *deliverable* `results/binds/b_e1_boundary.jsonl` · *evenings* 1

**S-22 · B-E2 conservation with the sink set** — phase 2 · prereq S-12 · independent-of S-21, S-23 to S-28 · parallel-safe · alias ledger P2.3
*build* $q^{({\rm sink})}+q^{(0)}+\sum_k q^{(k)}=\mathbf 1$ on $T$, printed with its residual on every batch **including when it fails** (V-23). Plants: drop BOS from every set and the solve must **raise** ($\rho(Q)=1.000000$, $\det(I-Q)=0.0$), never return a number; a set after the query gives $q=0.0$ exactly and the draw is discarded by census line 4. The struck plant — "drop the goal set, so $\max_k q^{(k)}\ge1/K$" — is a V-3 of the conservation row once the sink is present; the degeneracy lemma is carried as J-D2's **proposition**, its plant being "BOS declared inside a constraint set".
*prove* J-L15 when it builds; until then DERIVED.
*measure* the residual per batch: $4.4\times10^{-16}$ (`RUN[SATURN]`), $[0.9999999999999993, 1.0]$ (`RUN[I]`), $[1.000000000000000, 1.000000000000001]$ (`RUN[P]`).
*PASS* residual at or below $10^{-12}$ on $100$ per cent; the BOS-drop raises; the after-query set discards. *KILL* a number returned on the BOS-drop (the solver silently regularised, V-16, and every committor number in the lane is suspect); residual above $10^{-8}$ on any declared draw (a second undeclared absorbing row).
*price* $0$ GPU-s · *mechanism* V-12, V-23, V-16, V-3 · *deliverable* `results/binds/b_e2_conservation.jsonl` · *evenings* 0.5

**S-23 · B-G1 the interventional re-solve, with the triangularity plant** — phase 2 · prereq S-17 · independent-of S-21, S-22, S-24 to S-28 · parallel-safe · aliases ledger P2.4, B-C6
*build* (a) $V\equiv\mathbf 1$ implies $\max|\Delta z|\le10^{-15}$ (the bar is not "bitwise": the exact zero is a code-path accident); (b) Gaussian $V$ implies $\max|\Delta z|=O(1)$; (c) Sherman-Morrison closed form against a re-solve at or below $10^{-12}$; (d-prime) the **triangularity plant** — a non-causal $P$ makes $\Delta z_{<i}\ne0$, with the solve route named (`solve_triangular` gives $\Delta z_{<i}=0.0$ and `torch.equal True`; an LU route gives $8.9\times10^{-16}$ and `False`), and a two-row edit breaks the rank-one formula at $O(1)$ and is repaired by Woodbury; (e) the displacement identity's residual at or below $10^{-12}$. The EMC feedback plant is **deleted**: for $\gamma<1$ the fixed point is unique on every $P$.
*prove* J-L8 and J-L14 when they build.
*measure* (a) $0.0$ (`RUN[SATURN]`), $1.1\times10^{-16}$ (`RUN[M]`); (b) $0.1096$, $0.363$, $1.127$; (c) $1.2\times10^{-15}$, $8.9\times10^{-16}$, $4.4\times10^{-16}$; (d-prime) $0.0761$, $0.0868$; (e) $1.03\times10^{-15}$, $1.2\times10^{-15}$, $1.36\times10^{-15}$.
*PASS* (a), (c), (e) within bars and (b), (d-prime) at $O(1)$. *KILL* (b) reading below $10^{-6}$ on Gaussian $V$ (an empty rejection region: the bind is struck and Proposition 7 drops to a remark); (d-prime) reading zero on a non-causal $P$ (the solver is masking, not solving, and every $\Delta z$ number in the lane is suspect).
*price* $0$ GPU-s · *mechanism* V-24, D-5, D-7, V-3 · *deliverable* `results/binds/b_g1_resolve.jsonl` · *evenings* 0.5

**S-24 · B-G2 the vector-metric plants** — phase 2 · prereq S-16 · independent-of S-21 to S-23, S-25 to S-28 · parallel-safe · alias ledger P2.5
*build* the journal carries the vector, a histogram and quantiles (the record's Q6 census read `0 of 40`); three plants on the metric itself: (i) the permuted oracle must **not** score $0$; (ii) the oracle plus $0.1\sigma$ must be preferred to the permutation; (iii) $-\Delta z$ must be distinguished from $\Delta z$.
*prove* nothing.
*measure* the three plant readings.
*PASS* `3/3`. *KILL* any plant passing strikes the metric: pooled first-order Wasserstein fails (i) and (ii) by $14.465\times$ (`READ V20_R15_THEORY_TABLE.md:221`), which is why it is refused up front.
*price* $0$ GPU-s · *mechanism* V-26, L-14, V-16 · *deliverable* `results/binds/b_g2_metric.jsonl` · *evenings* 0.5

**S-25 · B-H1 the committor identity on the environment chain and on the arm's $\hat Q$** — phase 2 · prereq S-02, S-12 · independent-of S-21 to S-24, S-26 to S-28 · parallel-safe · alias ledger P2.6
*build* on the environment chain, with BED-1 as the instrument: $q=(I-Q)^{-1}R\mathbf 1$ against `bed_1.committor` on the **real** sets; the harmonic residual; the Kirchhoff second route at $K=2$. On the arm's causal $\hat Q$: invertible if and only if $0\in\mathcal{A}$ (J-L10), a diagonal read $\rho(\hat Q)=\max_T\hat P_{ii}$. Plants: the must-fire perturbation; declaring $\mathcal{A}_k$ on the wrong set.
*prove* J-L10 and J-L16(c) when they build.
*measure* $0.0$ on the real sets ($A=[0]$, $B=[1]$, $|T|=9$, `RUN[coord]`); residual $1.0408340855860843\times10^{-17}$; the perturbation drives it to $10^{-6}$, ratio $9.6\times10^{10}$; Kirchhoff below $10^{-10}$ (`READ MATHEMATICS.md:542-600`); $\rho(\hat Q)=0.692660$ both ways.
*PASS* $0.0$ on the real sets and both plants at $O(1)$. *KILL* the residual not moving under the perturbation (it is reading a cached $q$, V-3, and every committor identity in the lane is suspect); the arm-side solve returning a number with BOS undeclared (V-16).
*price* $0$ GPU-s · *mechanism* D-2, V-3, V-16 · *deliverable* `results/binds/b_h1_committor.jsonl` · *evenings* 0.5

**S-26 · B-H2 the argmin identity and the static-task kill** — phase 2 · prereq S-14, S-15, S-16, R-06 · independent-of S-21 to S-25, S-27, S-28 · parallel-safe · aliases V-5, ledger P2.7, K-H2
*build* the argmin from the oracle tensor equals the label's argmin on $100$ per cent of admitted draws (an identity of the builder, declared V-3); the rejection region is the control pair — a 0-hop per-position MLP on the candidate's context row and a 1-hop softmax, matched at the recounted parameter count (R-15), McNemar on identical draws.
*prove* nothing.
*measure* the identity fraction; the 0-hop and 1-hop accuracies with Clopper-Pearson; the McNemar $p$.
*PASS* identity at $100$ per cent; the 0-hop MLP outside the interval half-width of the shape at $N=8$ **and** McNemar $p\le0.05$ against the 1-hop softmax. *KILL* (K-H2) the 0-hop MLP inside the half-width, or $p>0.05$: BED-S is a third static task and is struck before any number is quoted; what dies is every capability sentence on BED-S, and §5.10 tree C is the paper.
*price* $\approx 0.5$ GPU-min (two control arms, 8 seeds each, at or below $1.524$ s) `[FITTED]` · *mechanism* V-24, V-3, C8, V-10 · *deliverable* `results/binds/b_h2_argmin.jsonl` · *evenings* 1

**S-27 · B-K containment and segmentation, the float instances** — phase 2 · prereq none (the Lean grades are Phase 0's) · independent-of every card · parallel-safe · alias ledger P0.1 float half
*build* the float instances the `[M]` targets must match: corner 3 against $(I-A)^{-1}$ entrywise and its last row against `equilibrium_oracle`; segmentation zeros by `torch.equal`, never `allclose`, on a masked $P$. Plants: keeping the diagonal breaks $A^s=0$ (`Nilpotent.one_not_nilpotent`); a $-30$ logit leaves the block non-zero. The $10^{-300}$ plant is struck as float64 underflow — a path of at least 80 gates at $0.5$ annihilates exactly, so "nonzero" can read F0.
*prove* J-L6, J-L11 when they build.
*measure* $0.0$ entrywise, $6.217248937900877\times10^{-15}$ on the last row (`RUN[coord]`); `torch.equal` zeros; the two plants at $O(1)$; the census `3 of 3` on BED-M and `0 of N` on BED-S's softmax corner.
*PASS* the three identities and both plants. *KILL* (K-K) any `[M]` target not building by the Lean milestone is `[S]` and the paper cites only declarations that build; what dies is the tag, never the float instance.
*price* $0$ GPU-s · *mechanism* P-11, V-25, V-2, L-LEAN · *deliverable* `results/binds/b_k_containment.jsonl` · *evenings* 0.5

**S-28 · The front-door plant register** — phase 2 · prereq S-01, S-20 to S-27 · independent-of nothing in Phase 2 · not parallel-safe
*build* one table listing, per bind, the plant's `kind`, its `manifest_hash`, the journal path, the $O(1)$ number it produced and the verdict line that consumed it; plus a census of `kind`s over `results/**/*.jsonl` calibrated **both ways**, as the wing manifest was (one arm must read zero, the control must read above zero, `READ V20_R15_WING_MANIFEST.md:73-83`).
*prove* nothing.
*measure* the FOUND count per plant and the two calibration reads.
*PASS* every plant of S-20 to S-27 FOUND, and the two calibration reads correct. *KILL* any plant that exists only in a script's print (NAMED): that bind is inadmissible and its proposition drops to a remark (V-14: the control that validates the matcher and never the reach).
*price* $0$ GPU-s · *mechanism* V-14, V-7, P-1 · *deliverable* `docs/apparatus/PLANT_REGISTER.md` · *evenings* 0.5

**S-52 ★ · The capped run at seeds 2, 3, 7 on BED-M that the record priced four times and never took** — phase 2 · prereq none · independent-of every card in this plan · parallel-safe · aliases M-2.1, R-13, V-6
*build* `arm_pl` on `e3_t2` at the three NO-READING seeds ($1.113403$, $1.139404$, $1.152430$, with $\hat a_{\max}$ $20.31$, $49.66$, $285.07$, `READ V15_R1.md:177-188`) with the magnitude capped at $1.0$, the same thread lane and flag regime as the retake journal, `synchronize()` bracketing and randomised order. The prediction and its counter are filed first (S-60).
*prove* nothing.
*measure* `eval_nrmse` and $\hat a_{\max}$ per seed against the uncapped values, against the one-hop capability threshold $0.7071067811865476$ — which is a threshold and never a floor (C15).
*PASS* all three capped seeds cross the threshold: the divergence was the cause of NO READING, and the paper's negatives section says so with the number. *KILL* any capped seed still above $1.0$: the split is not the cap's, the account of R1 keeps "3 of 8 NO READING, mechanism unresolved", and the Q3/W3 ledger cell is TERMINAL as the record's last verdict graded it. Either way the shape's own sentences are untouched; what dies is a stale question, cheaply.
*price* $\approx 9.65$ s ($3\times1.884+4.0$) `[FITTED]` · *mechanism* M-6 (a partial run read as a verdict), D-6 (the repair written and never started), P-3, C17 · *deliverable* `results/bedm/capped_seeds_2_3_7.jsonl` · *evenings* 0.5

**N-01 · Add `solve_triangular` as the fourth determinism quantity to the device certificate** — phase 2 · prereq none · independent-of every card · parallel-safe · alias ledger C-1 (first half)
*build* a specification for one more entry beside `hop / forward / gradient` in `scripts/k_cert.py::determinism_at_64` (`READ scripts/k_cert.py:579-640`): a batched lower-triangular $M$ with unit diagonal and a float32 right-hand side, the forward solve and its backward through a mean-squared error, 8 repeats, both flag regimes, single stream, workspace pinned before process start. Journal fields `solve_fwd.{bitwise,max_abs,executable}`, `solve_bwd.{...}`, `documented_guarantee`, `trsm_dispatch`. The certificate line is fixed here: forward bitwise, backward bitwise, both flag regimes, torch-documented guarantee NONE, cuBLAS dispatch `[U]`, one box, one process, one stream.
*prove* nothing; a repeatability reading, never a guarantee (V-16 forbids reading silence as a pass).
*measure* $\max|\Delta|$ over 8 repeats per quantity, three outcomes distinguished (bitwise, drifting, not executable).
*PASS* $0.0$ forward and backward under both regimes, reproducing the cost session's reading. *KILL* any $\max|\Delta|>0$ under the flag: the shape's training cells are journalled `warn_only` like the scan arms and Ruling 1's bitwise bar is not claimed for training; a raised error on the backward: the same fallback and the "first gated wing under strict mode" sentence is deleted.
*price* $\approx1$ GPU-s `[RUN class]` · *mechanism* P-1, V-16, V-23 · *deliverable* `results/k_cert_local.json` new keys plus one line in `COSTS.md` §1.6 · *evenings* 1

**N-02 · Make the solve microbenchmark a producer** — phase 2 · prereq N-01 · independent-of S-2x, N-19 · parallel-safe · alias M-2.3, ledger C-1 (second half)
*build* a `solve_increment` block timing $PV$, solve plus $Pz$, and Neumann $K\in\{1,2,4,8,16\}$ forward and backward at $n\in\{2048,4096,8192\}$, $s=64$, $d=16$, $\gamma=0.5$, float32, synchronize bracketed, 2 warm-ups, median of 14; journal fields `pv_ms, solve_ms, neumann_ms[K], increment_ms, n, s, d, gamma, cublas_workspace, producer_cmd`, plus peak bytes above a reset baseline per route.
*prove* nothing.
*measure* the increment per $n$: the reference values are $1.041$, $2.312$, $4.542$ ms ($2.514-1.473$, $4.659-2.347$, $9.916-5.374$, `RUN[NEPTUNE]`).
*PASS* each within the clock spread of plus or minus 12 per cent, and one Neumann hop slower than the solve at every $n$ ($3.001$ against $2.514$ ms at $n=2048$). *KILL* the increment at $n=2048$ above $2.2\times$ the reference: every shape price in §5.7 is re-derived before any card past the arena runs; the solve slower than one hop: the Neumann route re-enters as a cost path and N-10 is promoted.
*price* $\approx1$ GPU-min DERIVED; the whole `k_cert` rerun is $523.9$ s `[MEASURED]` · *mechanism* P-1, P-8, M-3, M-8 · *deliverable* `results/k_cert_local.json` `solve_increment` block · *evenings* 1

**N-07 · The training step under strict mode: the identical-seed floor for the shape** — phase 2 · prereq N-01, S-20 · independent-of S-2x · parallel-safe
*build* the floor run in the record's pattern (6 identical-seed pairs, `READ V17_R4_RETAKE.md:259-286`) for the shape and the softmax control under `use_deterministic_algorithms(True)` with `warn_only=False`, journalling `deterministic_regime = strict`.
*prove* nothing.
*measure* the NRMSE delta on 6 of 6 pairs; whether the backward executes.
*PASS* $0.0$ bitwise on `6/6` for both arms. *KILL* the shape's backward raises: the "first gated wing under strict mode" sentence is deleted and the lane runs `warn_only` (Ruling 1); softmax raises and the shape does not: the sentence stands with the control's row printed beside it (V-23).
*price* $\approx45$ s (the record's six pairs cost $44.7$ s) · *mechanism* Ruling 1, M-10, V-23 · *deliverable* `results/shape_floor_strict.jsonl` · *evenings* 1

**S-40 · Z-EX the exact route: $\delta=0$, the only route the committor head ships on** — phase 2 · prereq S-20 · independent-of S-41 to S-44 · parallel-safe · alias ledger Z-EX, Z-COM
*build* `solve_triangular` for $z$ and $O$ at any $\gamma<1$ and for the committor at $\gamma=1$ on the arm's triangular $\hat Q$; determinism as **three outcomes** (bitwise, drifting, raises), with `cumsum` required to raise and a run reporting "pass" on an operator that raised refused (V-16). The committor head carries **no** truncation certificate: the Neumann $\delta$ is infinite at $\gamma=1$ and the sup-norm substitute reads $19.2$ at $K=4$ against a true error $2.5\times10^{-3}$ (`RUN[M]`); the Perron-weighted certificate needs a weight that is `NOT MEASURED` (S-72). The threshold form $\max_k\hat q^{(k)}+\delta\|V\|_\infty\le\delta_{\rm thr}$ certifies the **solve, never the model**: $\hat P\ne P_{\rm env}$ has no certificate and no move is "admitted" by it.
*prove* J-L10, J-L13 when they build.
*measure* $1.8\times10^{-15}$ against the dense inverse; the three-outcome table from N-01.
*PASS* the table filled and `route = solve_triangular` on every committor cell. *KILL* a committor cell journalled with `route = neumann_K`: no certificate exists for it and the cell is refused.
*price* $0$ GPU-s (the microbenchmark is N-01's) · *mechanism* M-8, V-16, L-CERT, V-17 · *deliverable* `docs/apparatus/CERT_EXACT_ROUTE.md` · *evenings* 0.5

**S-41 · Z-NEU the Neumann certificate in vector units, with the convergent plant** — phase 2 · prereq S-01 · independent-of S-40, S-42 to S-44 · parallel-safe · alias ledger Z-NEU, B-I, P2.8
*build* for $P\ge0$ row-stochastic including absorbing rows and $\gamma<1$: the matrix residual
**equals** $\gamma^{K+1}/(1-\gamma)$ bare and $\gamma^{K+1}$ under $\Pi_\gamma$ — attained, hence
**definitional** on the class (declared V-3; `meyer-2000-matrix` owns the tail). The bind is carried by the plant and the units: rows scaled to $1.5$ at $\gamma=0.6$, $K=2$ read $7.290$ against $0.540$ with $\rho(\gamma P)=0.90$, a **convergent** series; the coordinator's $\gamma=0.7$ plant ($119.37$ against $1.143$) is a divergent partial sum and is labelled so. The vector statement $\max|z_{\rm solve}-z_K|\le\delta\,\|V\|_\infty$ is printed with $\|V\|_\infty$ and with $1/(1-\hat\gamma)$ beside every $\delta$ (the record's own factor was undefined at all eight R1 seeds, `READ V15_R1.md:56`).
*prove* J-L12 when it builds.
*measure* equality to $10^{-15}$ at $K\in\{1,2,4,8,16\}$ with absorbing rows (`RUN[M]`, error minus bound in $[4.4\times10^{-16}, 6.7\times10^{-16}]$); the plant's ratio.
*PASS* equality at all five $K$; the plant fails by at least $10\times$; `V_inf` and `one_over_1mg` present on every Neumann cell. *KILL* (K-I) the vector bound exceeded on any of 1,024 drawn cells, or $\delta\,\|V\|_\infty\ge\mathrm{sd}(\text{label})$ (uninformative), or the row-sum identity presented as evidence about $P$ (V-3): the certificate sentence dies and the exact route is the only one.
*price* $\approx0.04$ GPU-min for the 1,024 passes DERIVED · *mechanism* V-3, V-10, V-17, V-24, L-CERT · *deliverable* `results/certs/z_neu.jsonl` · *evenings* 0.5

**S-44 · Z-F0 the segmentation dividend in pair-count units, a corpus property** — phase 2 · prereq S-27 · independent-of S-40 to S-43 · parallel-safe · aliases N-14, ledger Z-F0
*build* the dividend $D=s(s+1)/\sum_m L_m(L_m+1)$ printed with the corpus named and the segment-length distribution beside it; three non-transfer clauses printed with it: (i) on the softmax corner an exponential has no exact zero, so the dividend is $1\times$ unless a mask is declared (F0 by mask, never by gate); (ii) a cut makes the segment head a new undeclared absorbing state and severs the boundary sets behind it, so segmentation and the reach-avoid read coexist only if every segment carries its own sink, goal and constraint sets with the cut a registered dial; (iii) the dividend applies to the exact, Neumann and CSR routes alike and carries to **no other corpus** (V-22).
*prove* J-L6, J-L7 when they build.
*measure* BED-M's live-pair fraction $0.0322$ gives $31.06\times$ (`READ V16_ARM_SMPRIME.md:518-522`; the source line's pair counts disagree by a factor of two and the discrepancy is flagged, the fraction is what is used); the alternative $s^2/\sum L_m^2$ would read $58.5$ and is V-17; BED-S's softmax corner prints $1\times$ and "F0 silent".
*PASS* both readings with their corpora named. *KILL* a dividend quoted for a corpus without its census, or carried across corpora (V-22); a committor cell run across an undeclared cut.
*price* $0$ GPU-s · *mechanism* V-22, V-17, V-25, D-3, L-CERT · *deliverable* `docs/apparatus/CERT_F0_SEGMENTATION.md` · *evenings* 1

**S-60 · File the predictions and counters, hash the file, record the hash before the first arena cell** — phase 2 · prereq S-12, S-13, S-16, S-35's settings · independent-of S-2x binds · not parallel-safe with S-62 · alias V-1
*build* one file with the five bets of §5.5, each with its prediction, a counter of equal specificity, the SPLIT band where one exists and the deciding statistic; plus S-52's bet and the language-model $\gamma$ row left blank until an LM cell runs behind the author's yes. Under the record's calibration (7 of 8 optimistic, $p=0.0352$) the counter is the point estimate (D-CALIB-1) and no shrink factor is fitted (D-CALIB-3). The file's sha256 is written into `results/arena/PREDICTIONS.sha256` **before** any arena cell exists, and every arena cell's manifest carries `predictions_hash`.
*prove* nothing.
*measure* the count of rows with both halves signed against the count of rows: it must read `24/24`.
*PASS* the hash is recorded, every bet has a counter, and the MDE cells reference the realised-sd column S-62 will fill. *KILL* a bet without a counter (D-7); a hash recorded after the first cell's timestamp (M-2): that cell's numbers are not quotable.
*price* $0$ GPU-s · *mechanism* D-7, L-SIGN, M-2, M-7, L-FIRST · *deliverable* `docs/apparatus/PREDICTIONS_R16.md`, `results/arena/PREDICTIONS.sha256` · *evenings* 1

---

### PHASE 3 — the arena on BED-S at the design point

**Goal.** Run every arm on byte-identical draws at $t^\star=8$, $m=8$, $K=2$, $s=64$, $N=8$ seeds deduplicated, one thread lane, and let the realised paired sd fix the minimum detectable effect that every kill formula in Phase 4 refers to. Two sequencing rules bind: the **controls run before the shape** (M-2, order stated), and **no arm trains before Phase 0's gate is green** (L-LEAN).

**The gate.** Eight FOUND records per arm with the header's flag regime present; the realised paired sd printed and the MDE row filled by the same guarded bisection that reproduces $0.039827$ at sd $0.034451$. If the realised sd is at or above $2.18\times$ the pilot's $0.050146$ — the M-3 precedent, where $0.109199$ gives $\mathrm{MDE}_8=0.126238$ — nothing in §5.5 is falsifiable at $N=8$, $N$ is repriced from the realised sd **before any bet is scored** (K-P), and the paper files the bed and no capability number.

**S-30 · The depth-1 softmax control with the same head** — phase 3 · prereq S-12 · independent-of S-31 to S-35 · parallel-safe · alias M-3.1 (first half)
*build* the matched control at the recounted parameter count (Ruling 3), with the same $[m,K+2]$ head or vector readout as the shape; it is the per-row control (Obstruction 1, definitional) and the owner of the single-location ground (`duranthon-2026-softmax-advantage` Prop. 4.2, whose label model admits `0 of 3` record beds).
*prove* nothing. *measure* per seed: $\varphi$-NRMSE, harmonic residual with its two unit factors, argmin accuracy with its interval, $\hat\gamma$ and $\Lambda$ where applicable; the paired sd of every contrast.
*PASS* one journalled cell per bed with `params_per_arm` in the manifest. *KILL* a count differing from the shape's by more than $0.1$ per cent after the head is attached: the pair is not matched and "at matched parameters" may not be written of it.
*price* $1.524$ s per cell `[FITTED]`, $\approx16.2$ s for eight seeds plus fixed cost · *mechanism* `MISTAKES.md` D-1, Ruling 3, R-SKY · *deliverable* `results/arena/softmax_d1.jsonl` · *evenings* 0.5

**S-31 · The two ChaCAL arms and the sink-token arm** — phase 3 · prereq S-20, S-21 · independent-of S-30, S-32 to S-35 · parallel-safe · aliases M-3.3 (part), V-15, R-19 (part)
*build* **ChaCAL-diag**, the shape with $\mathcal{A}=\emptyset$ and the diagonal kept — the identity half of B-E1 is against this arm only; **ChaCAL-published**, with the diagonal removed inside the inverse (`fagnou-2024-chacal` Eq. 5, `[V-fetched]` from one HTML render and **re-read against the PDF before it is typeset**), a regime-N inverse inside a regime-S read, sub-stochastic (`RUN[J]` row sums $0.400$ to $0.765$), and the planted negative of the bitwise half; **ChaCAL-with-sink-token**, a column device standing in for the row condition, the C2 kill's control. The "same $\gamma$" is the shape's **trained** $\hat\gamma$, so the pair is sequential and the order is stated in the manifest (M-2 in potential form). The struck control — "ChaCAL must emit chance" — is a control that must fail by construction (V-2) and is deleted.
*prove* nothing.
*measure* the creditable contrasts of S-13, per seed.
*PASS* three FOUND arms with `diag_convention` declared on each; the shape's committor $\varphi$-NRMSE below ChaCAL-with-sink's by more than $\mathrm{MDE}_8$ on at least 6 of 8 seeds and its residual smaller by more than $2\times$. *KILL* (K-E1, the expensive death) ChaCAL-with-sink within $\mathrm{MDE}_8$ on at least 6 of 8 **and** residual within $2\times$: escalate once to $N=16$ (S-64), never report "within TOST" at $N=8$ (M-13). What dies if it holds at $N=16$: the boundary-row mechanism as a **capability**; the paper becomes a property paper about a published operator, and §5.10 tree B is its sentence.
*price* $1.680$ s per cell `[FITTED + RUN]`, $\approx52$ s for three arms at eight seeds · *mechanism* V-3, P-10, V-24, M-13, R-SKY · *deliverable* `results/arena/chacal_{diag,pub,sink}.jsonl` · *evenings* 1.5

**S-32 · The InfSA-style Neumann read at $K=16$, no boundaries** — phase 3 · prereq S-12, S-41 · independent-of S-30, S-31, S-33 to S-35 · parallel-safe
*build* the truncated read $\sum_{k\le16}(\gamma P)^kV$ with its printed $\delta_\Pi=\gamma^{17}$ and $\delta\,\|V\|_\infty$ (`roffo-2026-infsa` `[V]` owns the reading); the exactness contrast `shape - Neumann-16` is creditable.
*prove* nothing. *measure* the gap between the exact solve and the $K=16$ read against $\delta\,\|V\|_\infty$ on every draw of the cell.
*PASS* one FOUND cell with `route = neumann_K`, $K=16$ and `delta_vec` filled. *KILL* the exact solve not within $\delta\,\|V\|_\infty$ of the $K=16$ read on any draw: either the arm is not row-stochastic on that cell (S-42) or the certificate is in the wrong units (V-17), and the exactness contrast is withdrawn.
*price* $\approx5.12$ s per cell, $\approx44.9$ s at eight seeds, DERIVED from the $25.409$ ms hop microbenchmark · *mechanism* `MISTAKES.md` D-1, L-CERT, V-17 · *deliverable* `results/arena/infsa_k16.jsonl` · *evenings* 0.5

**S-33 · The cached-mixture arm on the two plant classes** — phase 3 · prereq S-17, S-23 · independent-of S-30 to S-32, S-34, S-35 · parallel-safe · aliases M-3.5, R-25, V-16, ledger P3.3
*build* $O_{\rm cached}=\hat P_{\rm base}(I-\gamma\hat P_{\rm base})^{-1}V_{\rm int}$, with $\hat P$ frozen from the un-intervened context and values from the intervened one — the operational form of a cached successor representation (`momennejad-2017-sr`, `russek-2017-predictive`, mechanism `[U]`). It must fail on the $P$-changing plants and pass on the $V$-only plants, or the interventional channel is not needed.
*prove* nothing. *measure* field cosine and residual per plant class, paired by seed.
*PASS* on $V$-only plants within one seed sd of the shape; on $P$-changing plants worse by more than one seed sd on at least 6 of 8. *KILL* (K-G1) within one seed sd on $P$-changing plants on at least 6 of 8: the re-solve is a per-row control wearing a name (the precedent is the round where a "signed" pivot arm was the unsigned arm renamed, `READ workdonenew.md:379`), component (g)'s re-solve sentence dies, and the consequence channel is repriced as a cost statement.
*price* $1.680$ s per cell, $\approx35$ s for the two batches · *mechanism* D-2, P-7, V-9 · *deliverable* `results/arena/cached_mixture.jsonl` · *evenings* 1

**S-34 · The argmin controls: 0-hop MLP, 1-hop softmax, majority, random, predict-the-mean** — phase 3 · prereq S-26, R-06 · independent-of S-30 to S-33, S-35 · parallel-safe
*build* five arms with their own `kind`s, McNemar paired on identical draws; `random` reads $1/m$ and `majority` reads $\max_a\hat\pi(a^\star)$ by construction (declared V-3, printed as the floor row).
*prove* nothing. *measure* accuracy with Clopper-Pearson per arm; the two by-construction arms against their formulas.
*PASS* five FOUND cells and the two formulas reproduced within their intervals. *KILL* as S-26: the 0-hop arm inside the shape's half-width strikes the bed.
*price* $\le1.524$ s per cell · *mechanism* C8, V-10, V-3 · *deliverable* `results/arena/argmin_controls.jsonl` · *evenings* 0.5

**S-35 · The three skylines, settings fixed before the arena; the depth-5 cell** — phase 3 · prereq S-12, R-15 · independent-of S-30 to S-34 · parallel-safe · aliases M-3.4, R-23, V-14, ledger K-1
*build* the $\lfloor\log_2 t^\star\rfloor+2$ softmax stack — depth $3/5/7$ at $t^\star=2/8/32$, chosen
**by analogy** with `sanford-2024-logdepth` Thm 4.2, whose task is not the committor (the reduction is NOT FOUND, `sweep_expressivity.md` §3.5); the wide constant-depth stack at width $n_{\rm nodes}$ (`yehudai-2025-depthwidth`), which at $d_{\rm model}=16<s=64$ has no matched-parameter instance and the table says so; the chain-of-thought decoder at $t^\star$ steps (`merrill-2024-cot`). A deeper stack computes the same resolvent by iteration (`wang-2024-incontext-td`, `xie-2026-softmax-rl`), so "beats softmax" is banned either way (R-SKY).
*prove* nothing. *measure* $\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ against $\mathrm{MDE}_8$.
*PASS* (Bet D's prediction) within $\mathrm{MDE}_8$: the honest control holds and the shape's separate claims are exactness, one-read cost and the boundary mechanism. *KILL* outside in either direction, sign logged: short by more than $\mathrm{MDE}_8$ says the bed needs something the hop construction does not supply; ahead by more than $\mathrm{MDE}_8$ says "one operator" carries no accuracy sentence at all. A depth chosen after the arena strikes the row (M-2).
*price* $\approx7.6$ s per cell `[ASSUMED linear in depth]`, $\approx64.8$ s at eight seeds; the other two skylines `NOT MEASURED` with their settings journalled · *mechanism* `MISTAKES.md` D-1, R-SKY, P-10, V-25, D-7 · *deliverable* `results/arena/skyline_depth5.jsonl`, `docs/apparatus/SKYLINE_SETTINGS.md` · *evenings* 1

**S-62 ★ · The eight-seed arena, one invocation, one thread lane** — phase 3 · prereq S-12 admitted, S-21 to S-28, S-30 to S-35, S-60, S-61 · independent-of nothing in Phase 3 · **not** parallel-safe · aliases M-3.2, R-18, N-06, V-9 to V-13 (read off it), ledger P3.1, S-4
*build* one invocation: `device = cuda`, `threads = 8`, `steps = 150`, $n_{\rm train}=2048$, $n_{\rm eval}=4096$, $s=64$, $d_{\rm model}=16$, the flag regime journalled on the header; seeds 0 to 7 deduplicated; arms shape, softmax-d1, ChaCAL-diag, ChaCAL-published, ChaCAL-sink, InfSA-16, cached-mixture, plus the depth-5 skyline; every cell carrying the S-01 manifest, `predictions_hash`, `void_contrasts` and the census block; the run order randomised and every timed region bracketed by two `synchronize()` calls (the record's timer had run order as its strongest correlate, $\rho=+0.7029$, `READ V20_R15_JOURNAL.md:53`).
*prove* nothing.
*measure* per arm and seed: $\varphi$-NRMSE, the harmonic residual with $\sigma_{\min}$ and the operator norm printed, argmin accuracy with its interval, the field cosine and magnitude ratio, the sign column, $\hat\gamma$ and $\Lambda$, $\|\hat P-P_{\rm env}\|_\infty$, $1/(1-\hat\gamma)$; then the
**realised paired sd** of every scored statistic and the MDE row.
*PASS* eight FOUND seeds per arm, the header's flag regime present, the MDE row filled, and the identical-seed floor re-read. *KILL* (K-P) realised sd above $2.18\times$ the pilot: reprice $N$ before any bet is scored; what dies is the falsifiability of every $N=8$ kill in §5.5, and the paper files the bed with no capability number. *KILL* (K-D2) $\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at least 6 of 8: the arm copied the environment and every contrast on this bed is a copy-versus-no-copy statement.
*price* $7\times8\times1.680+4.0\approx98$ s $\approx1.6$ GPU-min DERIVED, plus $\approx64.8$ s for the skyline `[ASSUMED]` and $\approx44.9$ s for InfSA-16 DERIVED, so $\approx3.3$ GPU-min in total · *mechanism* M-10, M-16, V-26, M-3, Ruling 1 · *deliverable* `results/arena/r16_t8_m8_k2.jsonl` · *evenings* 1

**S-51 · The first BED-K cell of the arm's shape ever run** — phase 3 · prereq S-01, M-2.0 · independent-of every BED-S card · parallel-safe · alias M-3.6, ledger P3.4
*build* `bed_k.build_delay` on its pinned generator, against the native control `hard_delay_attention`, which reproduces the delay to $(n-1)e^{-45}$ (`READ ceq/beds/bed_k.py:311-336`); zero cells of BED-K's shape have ever run (`READ V20_R15_LEAP_LEDGER.md:323`). The Hankel ceiling is $\mathrm{err}_1=0.9746794345=\sqrt{1-1/20}$ at $d=20$ (DERIVED).
*prove* the shape with $\gamma>0$ has **no** delay advantage by theorem (`V15Kernel.first_order_cannot_delay`); the reading decides only that it is not worse than one head at a fixed offset (R-SKY).
*measure* NRMSE against the ceiling, $N=8$.
*PASS* the shape and the native control within $\mathrm{MDE}_8$ of each other. *KILL* the shape short by more than $\mathrm{MDE}_8$ on at least 6 of 8: a fixed-offset head the resolvent cannot represent at matched parameters, filed as a negative; the shape claiming a delay advantage is refused by theorem.
*price* $\approx34$ s per pair · *mechanism* V-3, R-SKY, V-25 · *deliverable* `results/bedk/first_cell.jsonl` · *evenings* 1

**S-50 · BED-M as containment: the parity bind and the truncation-law must-fire** — phase 3 · prereq S-20 · independent-of S-51 to S-53 · parallel-safe
*build* BED-M scores a scalar at one position and its label is the last row of corner 3's own resolvent, so `shape - corner-3` is VOID by registration (and by theorem once J-L11 builds); BED-M stays as the host of S-20's parity bind and of the truncation-law must-fire $\sqrt{(t^\star-k)/t^\star}$ at $k=0,1,2$.
*prove* J-L11.
*measure* $1.000058$, $0.712039$, $0.0$ at $t^\star=2$ (`RUN[VENUS]`); the one-hop threshold $0.7071067811865476$ printed as a threshold, never a floor — `13 of 40` banked cells violate it (`RUN[WATSON]`).
*PASS* both readings journalled with `void_contrasts` set. *KILL* any BED-M cell carrying a capability number for the shape is struck at assembly.
*price* $\approx34$ s if the ladder is re-run · *mechanism* `MISTAKES.md` D-1, D-2, C15 · *deliverable* `results/bedm/containment.jsonl` · *evenings* 0.5

**M-3.7 · Vary the dials: the one-factor-at-a-time sweep** — phase 3 · prereq S-62 admitted and not killed · independent-of S-31 to S-35, S-51 · parallel-safe
*build* seven design points — the centre $(t^\star,K,m)=(8,2,8)$ plus $t^\star\in\{2,32\}$, $K\in\{3,4\}$, $m\in\{4,16\}$ — each a shape-plus-softmax pair with the census re-run per point; $t^\star$ is placed by the DAG's depth and read off the hop ladder, never by a kill rate (which collapses the label: the record watched a label's sd fall from $0.499989$ to $0.038445$, `READ scale/e4_harmonic.py:425-434`).
*prove* nothing. *measure* the S-62 statistics per point; the exact zero-information floor per $m$ ($0.75$, $0.875$, $0.9375$ at $m=4,8,16$).
*PASS* every point admitted by its own census and the argmin error moves with $t^\star$ (D-3: a dial that does not vary describes nothing). *KILL* a point that cannot be placed ($t^\star=32$ unreachable inside the causal window) is **recorded, never forced**; error flat across $t^\star$ means the dial is decoration and only the centre is reported.
*price* $\approx3.5$ to $4.0$ min DERIVED ($7\times$ one pair) · *mechanism* D-3, P-8, V-25, V-10 · *deliverable* `results/arena/dials.jsonl` · *evenings* 2

**M-3.8 · Open the vector-label lane on BED-M** — phase 3 · prereq S-20 · independent-of every BED-S card · parallel-safe
*build* a separate lane and journal, because a vector label voids the record's published softmax baseline (`READ MISTAKES.md:701-708`); the `vector_readout` plumbing exists (`READ scale/m3_quintuple.py:368,483`); the label is the full $z^\star$ from `equilibrium_oracle` with the last-coordinate discard removed; shape against softmax with the vector readout, $N=8$.
*prove* nothing. *measure* the position-matched NRMSE vector; the harmonic residual with the chain as $P_{\rm env}$; S-24's three metric plants.
*PASS* a journalled vector with its histogram and quantiles (the record's Q6 census read `0 of 40`). *KILL* the permuted oracle scoring zero on the chosen metric strikes the metric (V-26); `shape - corner-3` on this lane is VOID by registration.
*price* $\approx34$ s · *mechanism* `MISTAKES.md` D-1, M-1, V-26, D-2 · *deliverable* `results/bedm/vector_lane.jsonl` · *evenings* 1

**V-17 · Bet G: a candidate move costs one column, not one solve** — phase 3 · prereq S-20 · independent-of every other Phase-3 card · parallel-safe
*build* the microbenchmark: after one solve at $n=2048$, $s=64$, $d=16$, price $m=8$ query-side row clamps two ways — (i) the Sherman-Morrison column route (one forward-substitution column at $s^2/2$ MACs plus an $s\,d$ inner product) and (ii) a fresh re-solve per candidate; synchronize bracketed, median of 14 after 2 warm-ups.
*prove* `sherman_morrison_row` `[S]` (J-L14), a row clamp only.
*measure* the measured ratio of (i) to one solve.
*PASS* (prediction) route (i) at or below $0.5\times$ one solve for $m=8$, since $m/d=0.5$ at $(8,16)$. *KILL* (counter) at or above $8\times$: the consequence channel is repriced as $m$ solves in every table; SPLIT in $(0.5\times, 8\times)$ prints the band and no point.
*price* seconds · *mechanism* M-8, P-8, M-3 · *deliverable* `results/arena/move_price.json` · *evenings* 1

---

### PHASE 4 — verdicts, the remaining certificates, the dossier

**Goal.** Score every bet against the number frozen before the run, append the verdict without touching the prediction, and write the dossier whose brackets are all filled from journals.

**The gate.** `VERDICTS.jsonl` carries one line per bet with `predictions_hash`, a token in `{HOLDS, COUNTER, SPLIT, VOID}` and the deciding interval; the prediction file is byte-identical to its hash; the calibration column has a row per scored bet whether or not it flatters. The author then reads one of the three sentences of §5.10 and knows which paper he has.

**S-63 · Record the verdict append-only; append the calibration row; do not edit the prediction** — phase 4 · prereq S-62 · independent-of nothing · not parallel-safe · alias V-19
*build* the adjudicator's output appended to `results/arena/VERDICTS.jsonl`; a superseded cell gets a `supersede` marker, never a deletion (L-G2); the calibration row (checked, wrong, sign) appended to the cross-round column whether or not it flatters (D-CALIB-5); the sign of every miss logged. The adjudicator itself reads **only** journals: it refuses below 8 distinct seeds, deduplicates by seed (bit-identical duplicate rows exist in the record), reads the thread lane from the journal and never from the machine (cross-thread drift $2.345\times10^{-3}$; a margin below about $4.7\times10^{-3}$ is indefensible), computes the realised paired sd from the first eight seeds, refuses any contrast smaller than the $n=8$ cell of that sd (M-3), checks `predictions_hash` on every cell against the recorded sha256 and **refuses to run on a mismatch**, and never writes to the prediction file.
*prove* nothing.
*measure* the adjudicator's own plants: a journal with 7 seeds returns INSUFFICIENT; a duplicated seed is counted once; a tampered prediction file halts the run.
*PASS* the verdict line is present and the prediction file is byte-identical to its hash. *KILL* any edit to the prediction file after the hash (M-2), any verdict filed without its bet id, or any branch computing a verdict on a standard nobody registered: the verdict is struck.
*price* $0$ GPU-s · *mechanism* M-2, D-7, P-3, L-G2, M-9, V-5, M-10, M-16 · *deliverable* `results/arena/VERDICTS.jsonl`, `docs/apparatus/ADJUDICATOR_SPEC.md`, the calibration column · *evenings* 1.5

**S-64 · The E1 escalation to $N=16$, only under the rule** — phase 4 · prereq S-63 · independent-of S-65 · parallel-safe · aliases M-4.1, R-19 (kill half), ledger P4.1
*build* if ChaCAL-with-sink is within $\mathrm{MDE}_8$ of the shape on at least 6 of 8 **and** the residual ratio is within $2\times$, escalate the pair to $N=16$ (seeds 8 to 15, same lane, same hash); $\mathrm{MDE}_{16}$ from the $n=16$ column ($0.025820$ at sd $0.034451$). Never "within TOST" at $N=8$: at that count two bit-identical arms return NO VERDICT (M-13).
*prove* nothing. *measure* the contrast at $N=16$ against $\mathrm{MDE}_{16}$.
*PASS* separated at $N=16$ by more than $\mathrm{MDE}_{16}$: the boundary-row mechanism is creditable. *KILL* not separated at $N=16$: component (e) is a parameterisation of a column sink, the paper's delta collapses to "ChaCAL plus a certificate plus Lean containment plus a registered bed", and every later card runs under that sentence (§5.10 tree B).
*price* $\approx44$ s ($3\times8\times1.681+4$) DERIVED, up to $\approx1.1$ GPU-min if two arms are re-run in full · *mechanism* M-13, M-7, D-7, M-9 · *deliverable* appended to `results/arena/r16_t8_m8_k2.jsonl` · *evenings* 0.5

**S-65 · The TOST parity half on $\mathcal{A}=\emptyset$ only** — phase 4 · prereq S-63, R-21 MOVED (otherwise the half is softmax against softmax) · independent-of S-64 · parallel-safe · aliases M-4.4, R-22, V-18, ledger P4.4
*build* the only contrast two one-sided tests decide: shape at $\hat\gamma$ against ChaCAL-diag at the same $\hat\gamma$ with $\mathcal{A}=\emptyset$ (`schuirmann-1987-tost` `[V]`); the margin $0.5\sigma$ is fixed **here** (M-2); the realised paired sd chooses the tier — paired at $N=36$ if the paired sd equals $\sigma$ (power $0.8014$, Monte Carlo with 200,000 draws), two-sample at $N=70$ if pairing buys nothing ($0.7975$ at 69, $0.8049$ at 70, half-width $0.2799\sigma$). The sequential anytime-valid form is NOT FOUND at `[V]`, so fixed-$N$ Schuirmann stands.
*prove* nothing. *measure* the realised paired sd over $\sigma$; the 90 per cent interval against plus or minus $0.5\sigma$.
*PASS* the interval inside the margin: the identity half of B-E1 is confirmed by equivalence, not only by bitwise identity at construction. *KILL* the interval outside: the "same operator" sentence is withdrawn at trained $\hat\gamma$ and narrowed to the $\gamma=0$ bind; and any "parity" sentence resting on $N=8$ is struck regardless (M-13).
*price* $119$ s paired at $N=36$ or $228$ s two-sample at $N=70$ DERIVED, both under $4$ GPU-min · *mechanism* M-13, M-9, M-2, V-5 · *deliverable* `results/arena/tost_parity.jsonl` · *evenings* 1

**R-21 · Score the $\gamma$ likelihood-ratio verdict, the ablation and the mirror line** — phase 4 · prereq S-61, S-62 · independent-of S-64, S-65 · parallel-safe · aliases M-4.3 (verdict half), V-12, ledger P4.3
*build* nothing new; S-61's instrument applied to the eight seeds.
*prove* nothing. *measure* $\Lambda$ per seed; the MOVED count; the ablation $(I-\hat\gamma\hat P)^{-1}\to I$ at trained weights in seed-sd units; the support of $\hat\gamma$.
*PASS* (Bet C) MOVED on at least 6 of 8 and the ablation moving NRMSE by more than one seed sd. *KILL* PINNED ($\Lambda\le2.7055$) on at least 6 of 8: the arm is softmax wearing a name on that bed, the horizon-dial sentence retires to a definition, ChaCAL-with-sink at $\hat\gamma=0$ is unrunnable so S-64 is void, and the LM $\gamma$ claim is retired before it is made. Mirror kill: $\hat\gamma>0.99$ on at least 6 of 8 prints $1/(1-\hat\gamma)>100$ and the certificate is vacuous on the $z$ and $\Delta z$ channel. The committor head is untouched either way: it reads at $\gamma=1$ on $\hat Q$.
*price* $0$ GPU-s (a journal read); the ablation $\le17.4$ s · *mechanism* Ruling 10-prime, V-9, M-20, V-17, P-8 · *deliverable* `docs/plan/GAMMA_VERDICT.md` · *evenings* 1

**M-4.2 · Score Bets A, B and E from the journal** — phase 4 · prereq S-62 · independent-of S-64, S-65, R-21 · parallel-safe · aliases V-10, V-11, V-13 (scoring halves)
*build* nothing new; the three deciding numbers with their frozen bands: the paired cosine difference against $\mathrm{MDE}_8$ computed on the **cosine's own** realised sd (V-17, never the NRMSE's); the argmin interval against the exact restricted-view floor, with the straddle as SPLIT; the harmonic residual ratio with $\ge10$ confirming, $\le2$ refuting and $(2,10)$ SPLIT, both unit factors printed.
*prove* nothing. *measure* as above, paired by seed and draw. *PASS/KILL* exactly as filed in §5.5 before the first cell. A ratio of at least $10$ at **unequal** marginal error is a statement about which mode the control mislearns, never about joint consistency, and is reported as such (`judge/sec_obstructions.md` §5.1).
*price* $0$ GPU-s · *mechanism* D-7, L-SIGN, V-26, V-17, M-2 · *deliverable* the calibration column of `docs/CALIBRATION.md` · *evenings* 1

**S-42 · Z-BETA the row-sum guard on trained cells** — phase 4 · prereq S-62 · independent-of S-63 to S-65 · parallel-safe · aliases ledger X-12, Z-BETA
*build* census lines 12 and 13 on every trained cell: the measured support of $\hat\beta$ and of $\mathrm{rowsum}(\hat P)$; at $\hat\beta\ne1$ rows sum $1.31$ to $10.29$ (`READ V16_ARM_SMPRIME.md:28-32`), where no row-stochastic theorem applies and the certificate field is written `refused`, never a number.
*prove* nothing. *measure* the two supports per cell.
*PASS* every cell with a maximum row sum above $1+10^{-9}$ has `delta_vec = refused`. *KILL* a $\delta$ printed on such a cell (V-25): that certificate is struck and every sentence resting on it is withdrawn.
*price* $0$ GPU-s · *mechanism* V-25, V-3 · *deliverable* `docs/apparatus/CERT_ROWSUM_GUARD.md` · *evenings* 0.5

**M-4.5 · File the leap dossier and the two honest sentences** — phase 4 · prereq S-63, S-64, S-65, R-21, M-4.2 · independent-of nothing · not parallel-safe
*build* the Phase-E template with the beds replaced (`READ CEQ_V20_R15_CONTRACT.md:153-169`): candidate built with identity binds, manifests, certificates, parameter-matched; verdict by the registered clauses; purge; prognosis. Both sentences of §5.10 with every bracket filled from a journal row or left `NOT MEASURED`.
*prove* nothing. *measure* the count of brackets filled from `results/` rows against the count of brackets.
*PASS* every number in the dossier points at a `results/` row. *KILL* a bracket filled from prose (P-1, P-2): the dossier is not filed and the paper says which bracket is empty.
*price* $0$ GPU-s · *mechanism* P-1, P-3, D-7, L-G2 · *deliverable* `docs/CEQ_SHAPE_DOSSIER.md` · *evenings* 1

---

### PHASE 5 — the cost law, memory, and the kernels that do not exist

**Goal.** Make every price at $s>64$ a measured law rather than an assertion, decide whether the serial solve is the shipped route, and write the specifications for the four instruments the paper marks `NOT MEASURED` so that a later evening can build them. Nothing in this phase is on the critical path to a verdict.

**The gate (K-9).** No price is quoted at any $s>64$ until S-66 runs with two `synchronize()` calls and a randomised order, and the fitted exponent's confidence interval is printed. If the interval excludes $2$ toward $3$, the serial solve is not the shipped route past $s=256$, N-11 moves onto the critical path ahead of every long-context bed, and every long-context price becomes a band with direction.

**S-66 · The synchronised, order-randomised $s$-sweep** — phase 5 · prereq N-02, M-2.0 · independent-of every Phase-3 and Phase-4 card · parallel-safe · aliases M-5.1, N-08, R-14, V-21, ledger P5.1
*build* the harness edits the record priced and never made: a sequence-length flag (the constant lives at `scripts/v15_r1.py:137` and no `seq_len` occurs, `RUN[MERCURY] grep`), one substitution, two `torch.cuda.synchronize()` calls, **and** a randomised or blocked cell order — the fifth edit no office ever priced. Points $(s,n)\in\{(64,2048),(256,2048),(1024,128),(4096,8)\}$, with $n$ declared per $s$ from §5.7's residency model; the dense operator at $n=2048$, $s=4096$ would be $137$ GB against $7.996$ GiB and is refused.
*prove* nothing. *measure* seconds per step and peak reserved bytes per (arm, $s$, $n$), $N=8$; the fitted exponent in $s$ with its interval; the Spearman correlation of seconds against run index.
*PASS* the exponent's interval contains $2$ and excludes $3$, and the run-order correlation is insignificant (the record's was $+0.7029$, $p=0.0024$). *KILL* (K-9) the interval excludes $2$ toward $3$; the solve does not beat the quadratic attention product at the $s$ where the paper claims it; or
**any price was quoted before this ran** — those prices are struck.
*price* the record's band $206$ to $537$ GPU-s, band only, no point · *mechanism* D-3, P-8, C17, M-3, K-9 · *deliverable* `results/cost/s_sweep.jsonl`, `docs/COST_LAW.md` · *evenings* 3

**N-09 · Where depth $s$ becomes visible: the solve-to-product ratio across $s$** — phase 5 · prereq S-66 · independent-of N-10 · parallel-safe
*build* from S-66's journal, the ratio of solve time to $PV$ time per $s$; the MAC prediction is plus 50 per cent on causal MACs ($3s^2d/2$ against $s^2d$, DERIVED), depth $s$ against depth 1.
*prove* nothing. *measure* the ratio at the four $s$.
*PASS* ratio at or below $1.5$ at $s\le256$ (arithmetic-bound). *KILL* ratio above $3$ at any $s\le1024$: the triangular-solve latency chain dominates, the chunked solve (N-11) is required before any long-context row, and the "plus 50 per cent" sentence is restricted to the $s$ where it was read.
*price* inside S-66 · *mechanism* M-3, P-8 · *deliverable* `docs/COST_LAW.md` depth row · *evenings* 1

**N-10 · The Neumann crossover, if any, across $s$** — phase 5 · prereq S-66 · independent-of N-09 · parallel-safe
*build* hop timings for $K\in\{1,2,4,8,16\}$ per $s$; the hops needed for $\delta=10^{-6}$ are $K\ge\log(\delta(1-\gamma))/\log\gamma-1$, which reads $19.9$ at $\gamma=0.5$ and $152$ at $\gamma=0.9$ — the cost section printed $129$ at $\gamma=0.9$ and the discrepancy is **recorded**, the formula being the deliverable.
*prove* nothing; the certificate is J-L12's, attained on the class and declared definitional.
*measure* the smallest $K$ at which $K$ hops beat the solve, per $s$.
*PASS* no such $K$ at any $s$: "truncation never wins" holds in wall-clock as in MACs. *KILL* a crossover at some $s$: the Neumann route is a cost path there, its $\delta\,\|V\|_\infty$ is printed with $1/(1-\hat\gamma)$ beside it, and the F1 mask certificate wakes on that $s$.
*price* inside S-66 · *mechanism* L-CERT, V-17, V-3, P-8 · *deliverable* `docs/COST_LAW.md` crossover row · *evenings* 1

**M-5.2 · Refit the cost law with $s$ as a variable and re-price every card** — phase 5 · prereq S-66 · independent-of M-5.3 · parallel-safe
*build* $\text{s/step}=a\,n^{b}s^{c}$ per arm with its R-squared and the interval on $c$; the increment law; the memory operator constant re-solved against the allocator, retiring the `[ASSUMED]` band.
*prove* nothing. *measure* R-squared per law; which `[ASSUMED]` tags the fit closes.
*PASS* every shape price carries a fitted law with R-squared at least $0.99$ and no `[ASSUMED]` remains on a Phase-3 or Phase-4 card. *KILL* R-squared below $0.99$ at any $s$: that $s$ is quoted as a measured point, never as a law — the record's own gate refused points fitted through swap (`READ COSTS.md:76-79`).
*price* $0$ GPU-s · *mechanism* M-8, P-8, V-22, P-1 · *deliverable* `docs/COST_LAW.md`, `docs/PRICE_LEDGER.md` revised with supersede markers · *evenings* 1

**N-18 · Re-solve the memory operator constant for the shape** — phase 5 · prereq S-20 · independent-of S-66 · parallel-safe
*build* the sizing sweep at the module's own settings over sequence lengths 128 to 2048, forward and backward, peak allocator bytes, run on the shape at the softmax corner; a two-parameter least squares separating the residual and operator constants — the record's $17.874$ and $3.823$ at R-squared $0.996373$ were fitted on the **signed** arm, so carrying $3.823$ to the shape is M-8.
*prove* nothing. *measure* the two constants for the shape with their R-squared, and the shape-to- softmax byte ratio per sequence length.
*PASS* R-squared at least $0.99$ and the ratio growing with $s$ (the class is quadratic in $s$, as $1.44\times$ to $8.06\times$ was for the signed arm). *KILL* measured over predicted outside $[0.9,1.1]$: the module's formula is declared wrong for this arm, as it was for the complex arm, and the shape's memory is quoted only from measurement.
*price* $\approx9$ GPU-min DERIVED · *mechanism* M-8, V-22, P-8 · *deliverable* `results/k_cert_local.json` memory law, `COSTS.md` §1.2 · *evenings* 2

**N-19 ★ · The bf16 gap: does `solve_triangular` have a bf16 CUDA path** — phase 5 · prereq none · independent-of every card · parallel-safe
*build* a one-line attempt on bf16 CUDA operands under autocast, recorded as executable or raising — the claim "no bf16 path in torch 2.5.1" is `[U]` and is settled here; then the bytes-per-element table for the shape under autocast beside the record's (operator $3.341$ B per element at R-squared $0.999830$; residual $2.383$ against the module's $2.2$, wrong by $8.3$ per cent in the optimistic direction, `READ COSTS.md:91-99`).
*prove* nothing. *measure* executability; the operator term for the shape under autocast.
*PASS* (the expected reading) the solve raises or upcasts and the shape's operand stays at $4.0$ bytes per element, printed as its own row. *KILL* (of the `[U]` sentence) a bf16 path exists: the row is refitted and the $4.0$ sentence is deleted.
*price* seconds · *mechanism* P-3, V-22, M-8 · *deliverable* `COSTS.md` §1.2 autocast row · *evenings* 1

**N-20 · The $n$ that does not fit: residency of the shape at large $n$** — phase 5 · prereq N-18 · independent-of S-66 · parallel-safe
*build* the residency table for the shape: the workhorse arm reserves $10.578$ GiB at $n=16384$ against $7.996$ GiB and pages over PCIe, which is why Ruling 8 dropped that reading and why the throughput law was fitted on three points rather than five; the shape on the softmax corner inherits the softmax rows (resident to $n=32768$ at $4.908$ GiB reserved) plus $8$ MiB per $2048$ for $z$ plus the solve's transients.
*prove* nothing. *measure* allocated and reserved GiB per $n$ under a training loop with synchronize.
*PASS* resident at $n=16384$. *KILL* reserved above $7.996$ GiB at $n=16384$: that reading stays dropped for the shape as for the arm, and any law point fitted through swap is refused (P-8).
*price* $\approx3$ GPU-min DERIVED · *mechanism* P-8, Ruling 8, V-22 · *deliverable* `COSTS.md` §1.3 · *evenings* 1

**N-21 · The sizing model's under-prediction for complex arms, and what the corner-3 base inherits** — phase 5 · prereq N-18 · independent-of N-20 · parallel-safe
*build* the complex-arm audit as a card: measured over predicted $1.842$, operator constant $7.50$ at 8 bytes per element ($4.29\times$ softmax's, not the $2\times$ a "complex is two floats" argument gives), the residual constant NOT IDENTIFIED at $s=64$. The shape on the corner-3 base inherits this and carries **no certificate** ($\hat\beta\ne1$ rows sum $1.31$ to $10.29$). The card either re-solves the complex constants with a sweep or declares that base uncertified for sizing.
*prove* nothing. *measure* measured over predicted across sequence lengths; the operator constant.
*PASS* constants re-solved with R-squared at least $0.99$ and the ratio stated per length. *KILL* R-squared below $0.99$ or the ratio outside $[0.9,1.1]$: the corner-3 base is journalled "uncertified for sizing" and no long-context card runs on it.
*price* $\approx5$ GPU-min DERIVED (the corner-3 cell is about $9.6\times$ softmax's) · *mechanism* M-8, V-22, P-8, V-25 · *deliverable* `COSTS.md` §1.2 complex row, `MODEL_CARD.md` limits line · *evenings* 1

**M-5.3 · Spend the 9,600-step ladder on a survivor only** — phase 5 · prereq S-63 to S-65 with the shape alive on at least one creditable contrast · independent-of S-66, M-5.2 · not parallel-safe
*build* the ladder (64 times the steps) for the surviving pair only.
*prove* nothing. *measure* the same statistics at the ladder's end; the crossing and residual curves.
*PASS* the $N=8$ verdict survives the ladder within its own $\mathrm{MDE}_8$. *KILL* a verdict that reverses at the ladder is filed as **the ladder's**, and the 150-step cell is marked a pilot (M-3); what dies is the 150-step sentence, not the bed.
*price* $32$ to $36$ min per pair `[FITTED]` at 64 times the law · *mechanism* M-3, M-6, P-8 · *deliverable* `results/arena/ladder.jsonl` · *evenings* 1

**N-11 · Specify and time the chunked block-triangular solve** — phase 5 · prereq N-09 · independent-of N-13, N-18 · parallel-safe · alias M-5.4
*build* the recursion $M_{kk}z_k=v_k-\gamma\sum_{l<k}P_{kl}z_l$ with $s/C$ diagonal triangular solves and one product per chunk: about $s^2d/2+sCd/2+sC^2/3$ MACs, depth $s/C$, retained blocks $sC$ (DERIVED). Invariant: parity against the serial solve at $10^{-6}$ in float32; journal `route = chunked` with $C$. The intra-chunk primitive is DeltaNet's transform pattern and is **owned**; the delta is stated narrowly: no finite-state dual exists for a softmax $P$ (`hu-2025-ssdtheory` `[V]`), so the inter-chunk term is a genuine product and the cost stays quadratic **because** the shape keeps softmax bitwise at $\gamma=0$.
*prove* nothing new; `resolvent_fromBlocks` `[M]` (J-L6) is the block identity.
*measure* parity; wall-clock at $s\in\{1024,4096\}$ against the serial solve for $C\in\{32,64,128\}$.
*PASS* parity at or below $10^{-6}$ at every $C$, and at $s=4096$ wall-clock at or below the serial solve. *KILL* slower at every $C$: the serial route is the shipped one and this card closes as NOT NEEDED on this device; parity failure: the kernel is refused (L-CERT — an uncertified route ships no $\delta$).
*price* `NOT MEASURED, needs a chunked kernel`; first timing is minutes once built `[ASSUMED]` · *mechanism* M-8, M-3, P-4, P-7 · *deliverable* `docs/CEQ_CHUNKED_SOLVE.md` · *evenings* 3

**N-12 · The recompute backward: leaving the quadratic memory class** — phase 5 · prereq N-11, N-18 · independent-of N-13 · parallel-safe
*build* the backward of N-11 rebuilding off-diagonal tiles from the query and key projections instead of retaining $P$; the adjoint of the solve is one upper-triangular solve plus an outer product (DERIVED); retained memory linear in $s$ plus $z$.
*prove* nothing. *measure* peak bytes against sequence length; gradient parity against autograd through the retained route at $10^{-5}$ in float32.
*PASS* peak bytes grow linearly within the sizing law's R-squared and parity holds. *KILL* bytes still quadratic: the recompute is not reached, the shape stays in the operator class, and the resident $n$ per $s$ from N-20 is the hard limit of every long-context card.
*price* `NOT MEASURED, needs the kernel` · *mechanism* P-4, V-22, P-8 · *deliverable* `docs/CEQ_CHUNKED_SOLVE.md` backward section · *evenings* 2

**N-13 · The published blockwise subquadratic evaluation as the occupied control** — phase 5 · prereq N-09 · independent-of N-11 · parallel-safe
*build* the control arm evaluating the same operator as `zhao-2026-structuredsparse` `[V]` does — exact triangular solves on diagonal tiles with cross-block interaction through a reduced system — as a
**fellow** at $s\in\{1024,4096\}$ with $n$ per $s$; its own $\delta$ is unprinted in the source and is therefore **measured here** where a dense control fits, and `NOT MEASURED` where it does not.
*prove* nothing; the shape's exact block structure is J-L6's F0 and holds only on gated corners or masked $P$.
*measure* visited-tile fraction, wall-clock, $\varphi$-NRMSE, harmonic residual, the measured $\delta$.
*PASS* the blockwise read within $\mathrm{MDE}_8$ of the dense read at lower latency, reproducing the source's band in kind. *KILL* the shape's F0-segmented solve not better than the blockwise control by $\mathrm{MDE}_8$ at matched visited tiles: the subquadratic path is theirs, cited, and the shape claims only exactness beside it — never "beats".
*price* `NOT MEASURED, needs the blockwise control implemented` · *mechanism* R-SKY, `MISTAKES.md` D-1, V-9, P-8, L-CERT · *deliverable* `results/cost/blockwise_control.jsonl` · *evenings* 2

**N-15 · The CSR two-stage path with the union certificate; the F1 mask bind wakes** — phase 5 · prereq N-03, N-10, S-41 · independent-of N-11, N-13 · parallel-safe · aliases S-43, S-70, V-20, R-10, ledger P5.2, Z-F1
*build* stage 1, the schedule as data: a causal list of tiles per query block, with the do-nothing zero-dimensional salience schedule **always entered** (`READ THEORY.md:162-165`); stage 2, Neumann on the sparse $P$ with the union certificate in vector units $(\varepsilon/(1-\gamma)+\delta_\Pi)\,\|V\|_\infty$ — the resolvent amplifies dropped row mass by $1/(1-\gamma)$ — and the **exact** solve on an F1 mask refused because the inverse fills in along reachability. The two structural guards stay: an empty schedule row must not return a division by zero, and a negative block index must be clamped (`READ ceq/mz_kernel.py:13-21`).
*prove* `mask_amplification` `[S]`, `f1_cantelli_union` `[D]` (J-L12, J-L17).
*measure* on the shipped mask, the gap over 1,024 drawn cells against the printed bound with $\|V\|_\infty$ printed; the fill-in guard on a planted schedule.
*PASS* no exceedance on any of 1,024 draws, $\delta\,\|V\|_\infty$ below the label sd, and the guard fires. *KILL* (K-I) one exceedance: the mask is refused (L-CERT) and the exact solve — cheaper than one hop at $s=64$, $2.514$ against $3.001$ ms — is the path; $\delta\,\|V\|_\infty$ at or above the label sd: the certificate is uninformative and the row returns to dormant.
*price* $\approx2.6$ s at $s=64$ DERIVED; at $s=4096$ `NOT MEASURED` · *mechanism* L-CERT, V-10, V-17, V-24, V-9 · *deliverable* `docs/CEQ_CSR_PATH.md`, `results/certs/mask.jsonl` · *evenings* 2

**N-16 · The Mapper cover to tile quantiser** — phase 5 · prereq N-15, S-44 · independent-of N-17 · parallel-safe · aliases S-70 (part), M-6.3 T-2, V-23
*build* a cover of key positions from a lens (the committor or an influence score) with parameters fixed by the Reeb-estimator rule on a **held-out** relation (`carriere-2018-mapper-statistics` `[V]`), never on the bed (M-2); nerve edges quantised to contiguous tiles **before** they are a schedule — a gather-realised candidate set pays $2.58$ ms of index-select against $0.82$ ms of attention at 65,536 positions on this card (`READ ceq/multizoom.py:12-16`); the far field carries the only printed bound in the tree, the mean-pool coarsening bound (`READ ceq/multizoom.py:38-49`). Cluster covers as candidate builders are occupied and cited; the delta is the nerve of a lens cover feeding a certified resolvent, stated as such. A Mapper cover with a comparable $\delta$ is NOT FOUND.
*prove* nothing. *measure* visited-tile fraction and $\varphi$-NRMSE against **two** do-nothing controls at matched visited tiles.
*PASS* better than both by $\mathrm{MDE}_8$ with the union $\delta\,\|V\|_\infty$ below the label sd. *KILL* (K-M) not better than do-nothing, or than the blockwise control: the candidate builder is the merged salience schedule, Mapper adds nothing measurable, and the corpus's segment-length distribution is the only dividend left.
*price* `NOT MEASURED, needs the quantiser and N-15 at long $s$` · *mechanism* V-9, M-2, L-CERT, V-22, P-4 · *deliverable* `docs/CEQ_MAPPER_SCHEDULE.md` · *evenings* 2

**N-17 · The backward through the scheduled kernel: the forward-only limit** — phase 5 · prereq N-15 · independent-of N-16 · parallel-safe
*build* the adjoint of the scheduled attention — the same schedule read transposed, the online-softmax statistics saved per query block, and for the resolvent stage the upper-triangular adjoint of N-12 — with both structural guards preserved. Until it exists the scheduled path is inference-side (`READ THEORY.md:223-224`), and the published speedup on that path is the PR page's own number, not re-measured here.
*prove* nothing. *measure* gradient parity at $10^{-5}$ in float32 on a dense schedule; forward and backward wall-clock against dense attention at $s=4096$.
*PASS* parity holds and the guards fire on planted empty rows and negative indices. *KILL* parity fails: the scheduled path stays inference-only and every training row above $s=64$ runs the chunked solve or the dense route with $n$ per $s$.
*price* `NOT MEASURED, needs the kernel` · *mechanism* P-4, V-16, L-CERT · *deliverable* `docs/CEQ_CSR_PATH.md` backward section · *evenings* 3

**J-D5 · Specify the directed-stability $\delta$ for the influence barcode** — phase 5 · prereq J-D4's notation, J-L6 · independent-of every Lean card · parallel-safe · aliases S-71, M-6.3 T-1, ledger T-1, P5.4
*build* the barcode row is **killed** on BED-S — the influence graph at threshold zero is one weak component on $100$ per cent of softmax draws, and the tree's interleaving instrument consumes point clouds (`READ ceq/certs/topological.py:476-505`) — and survives only on gated corners with exact zeros. What is owed before any barcode is read: (i) the filtered object, the influence matrix $J_{ij}=\partial O_i/\partial V_j=(\Pi_\gamma)_{ij}\ge0$, a **directed** weighted graph and not a metric; (ii) the connectivity notion, weak components of the thresholded digraph (`chowdhury-2017-path-homology` counts them at degree 0); (iii) the dissimilarity transform $d_{ij}=-\log J_{ij}$ so a filtration of an asymmetric function is defined; (iv) the stability constant — `turner-2019-quasimetric-rips` gives bottleneck stability for asymmetric functions, and the symmetric-hypothesis stability theorems do **not** apply — the printed $\delta$ being that constant times the sup-norm perturbation, `NOT MEASURED, needs the constant evaluated on the shipped filtration`; (v) the null, row-permuted influence at the same logit scale, at least 200 permutations; (vi) the threshold-zero endpoint equal to the F0 segmentation count by `torch.equal`.
*prove* nothing. *measure* the six-part predicate written before any barcode is computed.
*PASS* the predicate is written first. *KILL* the measured barcode inside the null's 95 per cent band on at least 6 of 8 seeds: decoration; a non-integer or aliased component count: refusal (V-16).
*price* $0$ GPU-s for the spec; the instrument `NOT MEASURED` · *mechanism* M-15, M-2, V-16, V-3, V-25, P-4 · *deliverable* `docs/apparatus/DIGRAPH_BETA0_SPEC.md` · *evenings* 2

**S-72 · The Perron weight for a committor certificate** — phase 5 · prereq S-40 · independent-of N-1x, J-D5 · parallel-safe · aliases ledger P5.7, Z-COM
*build* the contraction certificate needs a weight $w$ with $Qw\le\rho w$ and $\rho<1$ for the
**oracle's** non-causal chain; on the arm's causal $\hat Q$ the diagonal read of J-L10 suffices and no weight is needed. Until $w$ is measured, every committor cell ships on the exact route only.
*prove* `isUnit_one_sub_of_perron` `[S]` (J-L15).
*measure* the weight, if it is ever computed; until then the sup-norm substitute is recorded as vacuous ($19.2$ at $K=4$ against a true error $2.5\times10^{-3}$).
*PASS* a weight with its two inequalities printed. *KILL* a truncated committor read journalled without $w$ is refused; what dies is any "certified committor at a truncation" sentence.
*price* `NOT MEASURED, needs the Perron weight` · *mechanism* V-17, V-10, L-CERT · *deliverable* `docs/apparatus/PERRON_WEIGHT.md` · *evenings* 1

**J-D8 · Attempt the hop-to-committor reduction, or file it NOT FOUND** — phase 5 · prereq J-D4 · independent-of every card · parallel-safe
*build* one page: given a $\mathrm{hop}_k$ instance, construct logits whose causal softmax puts mass $1-\eta$ on the pointer target, declare the $k$-th target set absorbing, and show the committor into it is at least $(1-\eta)^k$ while the committor into any other set is at most $1-(1-\eta)^k$, so the argmax over sets reads $\mathrm{hop}_k$. What blocks a theorem: the softmax row also puts mass on itself (J-L5) and on the sink (J-L2), so the walk can stall or fall; the construction must bound the stall mass, and that bound depends on the logit scale, which the arms **learn**.
*prove* either the reduction with its stall-mass bound, or NOT FOUND with the blocking step named.
*measure* nothing.
*PASS* either outcome written. *KILL* the reduction stated **without** the stall-mass bound (V-3, P-10); then Obstruction 2 remains a lineage argument with its vacuity lines, which is what the paper says today.
*price* $0$ GPU-s · *mechanism* V-3, D-2, P-10 · *deliverable* `docs/CEQ_SHAPE.md` §5.2 · *evenings* 2

---

### PHASE 6 — the envelope, the package, the Kaggle gate

**Goal.** Make the local half of the gate complete so that, if and when the author says yes, nothing is blocked on work an agent could have done locally.

**The gate.** Every local certificate line filled (N-01, N-02, N-18), the envelope's two plants firing, the package's smoke passing, the pinned digests re-asserting on regeneration; the open rulings listed by name beside the request. **The request is a notification to the author, never a launch.** The standing rule is the author's: no launch, upload or training start without an explicit yes.

**N-22 · The flight envelope as the run's guard, written for the shape lane locally** — phase 6 · prereq S-01 · independent-of every card · parallel-safe
*build* Ruling 6's clauses as fields every shape cell carries, plus the three run-guards the record already owns: the zero-step RED gate ($\mathrm{GATE\_TOL}=10^{-3}$; 30 of 30 shapes passed with worst margin $1.957\times10^{-3}$), the bar re-certification HALT line (delta over tolerance below 50 per cent; the record's worst was $8.58$ per cent with $5.83\times$ headroom), and the thread lane read from the journal and never from the machine.
*prove* nothing. *measure* two planted faults — an out-of-memory-sized shape and a bar drifted past the halt line — must abort and halt respectively, each with a journal row.
*PASS* both plants fire and an honest run passes both. *KILL* a plant that does not fire: the envelope is a condemning rule with no planted negative (V-15) and no launch request is made until it does.
*price* seconds · *mechanism* Ruling 6, V-15, V-16, M-16, D-4 · *deliverable* `docs/CEQ_ENVELOPE.md` · *evenings* 1

**N-23 · The package: replace the dead operator with the shape, match the count, keep the card honest** — phase 6 · prereq S-20, S-01 · independent-of S-66 to N-21 · parallel-safe
*build* the attention module becomes ChaCAL-diag with declared boundary rows and the manifest's `route` and `diag_convention` fields; the smoke script asserts parity at $\gamma=0$ against the package's own softmax attention; the parameter count is recounted per arm with the $\gamma$ scalar and any head included (Ruling 3: exact counts in every table header, never re-architect to close $0.032$ per cent); the model card is rewritten limits-first, with §5.10 tree B's sentence as its capability paragraph until a cell exists. No checkpoint ships.
*prove* nothing. *measure* parity, the counts, one forward at $s=64$ on CPU.
*PASS* parity `True`, the counts printed and within $0.032$ per cent, no checkpoint claimed. *KILL* parity fails: the package ships no shape; a card sentence with a bracket filled by anything but a FOUND cell is struck (P-1).
*price* $0$ GPU-s · *mechanism* P-3, Ruling 3, V-3, P-1 · *deliverable* the package edits and `MODEL_CARD.md` · *evenings* 2

**N-24 · The Kaggle gate, local half only** — phase 6 · prereq N-01, N-02, N-18, N-22, N-23 · independent-of nothing (it is the DAG's sink) · not parallel-safe · alias M-6.1
*build* the local half of Gate 0 only: the three bed pins carried as generator, seed and hash and regenerated in-notebook with the hash asserted (Ruling 7); the corpus pin with its split; the three unpinned sources whose hash cell prints and then **raises**; the code pin, which is the snapshot the run executes and may differ from the branch tip. The Kaggle certificate slot is empty and stays empty until the author's run; a threshold carried across the device boundary is V-22. The gate's own circularity — its first rows are blocked on numbers only the Kaggle run produces — is the author's ruling and is not taken here. The launch order is unchanged, and its only consumer in this programme is the LM $\gamma$ cell.
*prove* nothing. *measure* locally: that N-01, N-02 and N-18 have filled their certificate lines, that N-22's plants fire, that N-23's smoke passes, and that the pinned digests re-assert on regeneration.
*PASS* every local line filled and the open rulings listed by name beside the request. *KILL* any Kaggle number pooled with a local one (V-22); any launch, upload or training start without the author's explicit yes; a missing-pin raise bypassed.
*price* $0$ GPU-s locally; on Kaggle the certificate run is $\approx8.7$ GPU-min `[ASSUMED]` equal to the local $523.9$ s · *mechanism* V-22, P-1, Ruling 7, the Kaggle-yes rule · *deliverable* `docs/KAGGLE_GATE.md` · *evenings* 1

**S-73 · The LM $\gamma$ cell on the pinned slice, behind the author's explicit yes** — phase 6 · prereq S-61, S-63, R-21 MOVED, N-24, **the author's yes** · independent-of nothing · not parallel-safe · aliases M-6.2, V-22, ledger P5.8
*build* one LM pair — shape against ChaCAL at fixed $\gamma=0.9$ — on the pinned slice, with $\hat\gamma$ per seed under the boundary null; the language-model row of the predictions file is filled only then.
*prove* nothing. *measure* $\Lambda$ per seed; the perplexity delta against ChaCAL against $\mathrm{MDE}_8$.
*PASS* MOVED on at least 6 of 8. *KILL* PINNED on at least 6 of 8 — consistent with ChaCAL's own language-model result being worse than its baseline (digits `[U]`): the LM $\gamma$ claim is retired and $\gamma$ stays a bed-side dial.
*price* `NOT MEASURED` — no LM cell of the shape has ever run and the Kaggle certificate slot is empty · *mechanism* V-22, Ruling 10-prime, V-9, the Kaggle-yes rule · *deliverable* `results/lm/gamma.jsonl` after the yes · *evenings* 2 after the yes

---

## 5.4 The critical path, and the evenings count derived from it

**The path, as a chain of card ids with each card's evenings.** The binding constraint is L-LEAN: the arm may not be **trained** before its identity theorems are green, and the theorems S-20 rests on are J-L1 (parity) and J-L3 (the unit), reached through J-L0 and certified by J-L18. Everything else either runs beside that chain or is shorter than it.

$$ \underbrace{\text{J-L0}}_{1}\to\underbrace{\text{J-L3}}_{1}\to\underbrace{\text{J-L10}}_{1}\to \underbrace{\text{J-L18}}_{1}\to\underbrace{\text{S-20}}_{1}\to\underbrace{\text{S-21}}_{1}\to \underbrace{\text{S-26}}_{1}\to\underbrace{\text{S-60}}_{1}\to\underbrace{\text{S-62}}_{1}\to \underbrace{\text{S-63}}_{1.5}\to\underbrace{\text{S-64}}_{0.5}\to\underbrace{\text{S-65}}_{1}\to \underbrace{\text{M-4.5}}_{1}
$$

$$1+1+1+1+1+1+1+1+1+1.5+0.5+1+1 \;=\; \mathbf{13}\ \text{evenings.}$$

**The one branch that lengthens it.** The bed branch $\text{S-10}(0.5)\to\text{S-11}(2)\to\text{S-12}(1)\to\text{S-16}(1)=4.5$ must finish before S-21, and it runs beside the four-evening Lean gate, so it adds $0.5$: **13.5, called 14 evenings to a verdict and a filed dossier.** Add M-5.3 (the ladder on a survivor, 1) for **15**; add the cost-law branch $\text{S-66}(3)\to\text{M-5.2}(1)$ if a cost law is demanded before the ladder, for **19**.

**What runs beside it, and therefore costs no additional evening.** M-2.0 (build the arm, 3) — it must precede S-20 but is shorter than the Lean gate. The `[S]` Lean set J-L11 to J-L17 (9 evenings) — it changes the paper's **grades**, never the training gate. Every J-D derivation (8 evenings, all parallel-safe). Every Phase-1 card except the four on the branch above. S-52, S-51, M-3.7, M-3.8, V-17, N-01, N-02, N-07, N-19, S-40 to S-44. All of Phase 5's kernels, which have no prerequisite in the verdict chain at all.

**Reconciling the six planets' own critical-path claims.** Each planet counted the serial depth of
**its own slice**, and the differences are exactly the slices, not a disagreement about the DAG.

| planet | its claim | what it counted | where it overlaps this plan's 14 |
|---|---|---|---|
| JUPITER (Lean) | **11 evenings**: J-L0 → J-L3 → J-L10 → J-L13 → J-L12 → J-L15 → J-L16 → J-L18 | the chain to the full `[S]` set, not the training gate | its first four evenings (J-L0, J-L3, J-L10, J-L18) are the training gate and are **on** the merged path; the other seven are beside it |
| SATURN (instrument) | **11 evenings**: S-01 → S-11 → S-12 → S-14 → S-20 → S-21 → S-60 → S-61 → S-62 → S-63 → S-64 | the bed-and-verdict chain with no Lean gate in front of it | eight of its eleven are on the merged path; it omits the four Lean evenings and folds S-26 into S-14 |
| MERCURY (price) | **12 evenings** to the ladder, **15** with the cost law | M-1.1 → M-1.2 → M-1.4 → M-2.4 → M-3.1 → M-3.2 → M-3.3 → M-4.1 → M-4.4 → M-5.3 | the same bed-and-arena spine under different ids; its M-2.0 (3 evenings) is beside, as here |
| MARS (attack) | **7 evenings**: R-04 → R-07 → R-09 → R-18 → R-19 → R-22 | the attack chain only, with the Lean gate explicitly beside it | its six are a subset: the bed spec, one census attack, the binds, the first pair, the arena, the equivalence test |
| VENUS (prediction) | **9 evenings**: V-0 → V-1 → V-3 → V-5 → V-10 → V-15 → V-18 → V-19 | the prediction-filing and scoring chain | V-1 is S-60, V-10's pair is S-62, V-15 is S-31 plus S-64, V-18 is S-65, V-19 is S-63 |
| NEPTUNE (systems) | **18 evenings**: N-03 → N-05 → N-06 → N-08 → N-09 → N-11 → N-12 → N-15 → N-17 → N-16 | a serial depth through **kernels that do not exist**; ten of its eighteen are Phase-5 specifications | only N-05 (which is S-20) and N-06 (which is S-62's price) touch the merged path; the rest is off it by construction |

The six agree on the spine — bed specification, census, binds, first pair, arena, verdict — and differ only in what they put in front of it (JUPITER: Lean; NEPTUNE: kernels) and in whether they count the arm's construction as an evening. The merged number is **14 to a verdict**, and the iteration count of any loop that runs this plan is that depth, never an agent's choice (`CONTRACT.md` D-3).

---

## 5.5 The predictions ledger, with counters

Filed before any arena cell exists and hashed into `results/arena/PREDICTIONS.sha256` (S-60). Every row carries a counter of equal specificity (L-SIGN), the SPLIT band where one exists, and the single number that decides between them. Under the record's calibration — 9 checked, 9 adverse, 7 of 8 signed rows optimistic, one-sided sign test $p=0.0352$ — **the counter is the point estimate** (D-CALIB-1), the sign is the only output (D-CALIB-3), and a bare prediction is refused at the file rather than discounted (D-CALIB-2). $\mathrm{MDE}_8$ everywhere means the $n=8$ cell of the
**realised** paired sd, which BED-S has not produced; the figure $0.039827$ at sd $0.034451$ is the record's row-2 instance and is a placeholder until S-62 fills it (M-3).

| bet | prediction (the half that flatters) | counter (the point estimate) | SPLIT band | deciding number | card | price | mechanism |
|---|---|---|---|---|---|---|---|
| **A** consequence field | $\cos_{\rm shape}-\cos_{\rm softmax}>\mathrm{MDE}_8(\cos)$ on the paired test | within $\mathrm{MDE}_8(\cos)$: a per-row mixture learns the displacement coordinate by coordinate as well as the joint read | none (two-sided on the sign of the miss) | the paired cosine difference against the **cosine's own** realised sd | S-62, M-4.2 | inside the $\approx34$ s pair | `MISTAKES.md` D-1, V-26, V-17 |
| **B** argmin floor | $\mathrm{CP_{upper}}(\mathrm{err}_{\rm shape})<\mathrm{floor_{exact}}(k=1)$ at $N=8$ | $\mathrm{CP_{lower}}\ge\mathrm{floor_{exact}}(k=1)$: the joint read is no better than a one-hop window licenses | the straddle $\mathrm{CP_{lower}}<\mathrm{floor}\le\mathrm{CP_{upper}}$ (the R1 precedent straddled $0.7071$ with $[0.617075, 1.041227]$) | the interval against the floor | S-62, M-4.2 | inside the pair | L-FLOOR, V-10, M-2 |
| **C** the horizon dial | $\hat\gamma$ MOVED ($\Lambda>8.318$) on at least 6 of 8 seeds, and the ablation moves NRMSE by more than one seed sd | PINNED ($\Lambda\le2.7055$) on at least 6 of 8: softmax wearing a name on this bed | $3/8$ to $5/8$, or interval verdicts | $\Lambda$ per seed under the boundary null | S-61, R-21 | $0$ GPU-s plus a $\le17.4$ s ablation | Ruling 10-prime, V-9, M-20 |
| **D** the depth skyline | the depth-5 stack within $\mathrm{MDE}_8$ of the shape's argmin accuracy — the honest control holds | short by more than $\mathrm{MDE}_8$, sign logged: the bed needs something the hop construction does not supply | none | $\mathrm{acc}_{\rm shape}-\mathrm{acc}_{\rm sky}$ | S-35 | $\approx64.8$ s `[ASSUMED]` | `MISTAKES.md` D-1, R-SKY, P-10 |
| **E** joint consistency | $r_{\rm softmax}/r_{\rm shape}\ge10$ at marginal NRMSE within $\mathrm{MDE}_8$ | $\le2$: joint consistency is learnable by a per-row mixture and the untested sentence at `MATHEMATICS.md:106-127` is withdrawn | $(2,10)$ | the paired ratio with $\sigma_{\min}$ and the operator norm printed | S-62, M-4.2 | inside the pair | V-26, V-17, D-2 |
| **F** the boundary mechanism (K-E1) | the shape's committor $\varphi$-NRMSE below ChaCAL-with-sink's by more than $\mathrm{MDE}_8$ on at least 6 of 8, residual smaller by more than $2\times$ | within $\mathrm{MDE}_8$ **and** residual within $2\times$: component (e) is a parameterisation of a column sink | escalate once to $N=16$, never "within TOST" at $N=8$ | the paired $\varphi$-NRMSE difference and the residual ratio | S-31, S-64 | $\approx52$ s, escalation $\approx44$ s | V-24, M-13, M-7 |
| **G** the move price | the Sherman-Morrison column route at or below $0.5\times$ one solve for $m=8$ ($m/d=0.5$ at $(8,16)$) | at or above $8\times$: a re-solve per candidate is what ships | $(0.5\times, 8\times)$ | the measured ratio | V-17 | seconds | M-8, P-8, M-3 |
| **H** the cached mixture (K-G1) | the cached arm fails on $P$-changing plants by more than one seed sd on at least 6 of 8, and matches on $V$-only plants | within one seed sd on $P$-changing plants: the re-solve is a per-row control wearing a name | none | the two per-seed cosine gaps in seed-sd units | S-33 | $\approx35$ s | D-2, P-7, V-9 |
| **I** the capped seeds (the record's own open item) | all three capped seeds cross the one-hop threshold: divergence was the cause of NO READING | at most one crosses: the split is not the cap's and the mechanism stays unresolved | $2$ of $3$ | the crossing count out of three | S-52 | $\approx9.65$ s | M-6, D-6, P-3 |
| **J** the certificate (dormant) | zero exceedances on 1,024 draws and $\delta\,\|V\|_\infty$ below the label sd | one exceedance, or $\delta\,\|V\|_\infty$ at or above the label sd: the mask is refused | none | the exceedance count and the ratio | N-15 | $\approx2.6$ s when a mask exists | L-CERT, V-10, V-17 |
| **K** the Lean set | `10/10` `[M]` targets elaborate with the three-axiom set | at most `9/10`: a row resting on an unconfirmed Mathlib name is demoted to `[S]` and never cited as proved | none | the green count out of ten | J-L18 | $0$ GPU-s | P-11, L-LEAN |
| **L** the cost exponent | the fitted exponent's interval in $s$ contains $2$ and excludes $3$, and the run-order correlation is insignificant | the interval excludes $2$ toward $3$: depth $s$ is visible past $s=64$ and every long-context price is a band | none | the exponent interval | S-66 | $206$ to $537$ GPU-s, band only | D-3, P-8, C17 |
| **M** the LM dial | $\hat\gamma$ MOVED on at least 6 of 8 enwik8 seeds | PINNED on at least 6 of 8, consistent with the published operator's own language-model result being worse than its baseline | none | the MOVED count | S-73 | `NOT MEASURED`, behind the author's yes | V-22, Ruling 10-prime |

**The scoring rule.** One ledger, two sources: an `author` row is a card above; a `leap` row is any prediction inside a leap output, filed `[LEAP-UNTESTED]` with its counter and its cheapest killer, its instance RUN **before** the row is scored. Verdict tokens are `HOLDS`, `COUNTER`, `SPLIT`, `VOID`; a VOID row (the bed not admitted, or a prerequisite instrument failing its plants) is unscored, never a miss; a SPLIT row is scored wrong with the sign of the optimistic half. The one-sided sign test runs on the pooled column after at least eight scored rows and on the two sub-columns separately. Rows are appended, never edited; a corrected reading is a new row whose `supersedes` points at the old one (L-G2). A column that stays at or above $0.75$ optimistic after eight rows makes the counter the paper's **reported** estimate for every unscored card, not merely its planning estimate.

---

## 5.6 The standing attacks

Each is the cheapest run, with its threshold frozen here (M-2), that ends a claim sentence the judged sections permit. Three of them (R-07, R-08, R-24) were filed by no office before this plan and each has an $O(1)$ instance on the judge's draw, not a rate.

| attack | target claim it ends | firing number (frozen) | price | mechanism |
|---|---|---|---|---|
| **R-01** the `[M]` targets do not build | "machine-checked" for component (k) | any target not building by the Lean milestone is `[S]`; a `sorryAx` anywhere deletes the row | $0$ GPU-s | P-11, L-LEAN |
| **R-02** the manifest does not discriminate | every attribution in the lane | any one-unit flip of `beta/qk/g/gamma/boundary_sets` leaving `manifest_hash` fixed | $0$ GPU-s | L-2, V-16 |
| **R-03** an obstruction without its vacuity line | every "softmax cannot" sentence | any obstruction stated without its inequality; $0$ per cent of the bed admitted by the hypothesis | $0$ GPU-s | V-25, P-10 |
| **R-04** the bed dies at construction | every BED-S sentence | any label sd $=0$; any argmin class frequency outside $(0.05,0.95)$; disagreement $=0$; lines 1 to 4 below $100$ per cent | $0$ GPU-s | D-4, V-8, V-12 |
| **R-05** the label leaks at order 0 | the one-read claim on BED-S | honest probe at or above $0.5$; or the planted leak below $0.99$ (a blind detector) | $0$ GPU-s | M-21, V-24, V-7 |
| **R-06 / K-H2** the bed is a static task | every capability sentence on BED-S | the 0-hop MLP inside the shape's interval half-width at $N=8$, or McNemar $p>0.05$ against 1-hop softmax | $\approx0.5$ GPU-min | C8, V-10 |
| **R-07 ★** the Chebyshev rule is minimised by falling off the prompt | the Chebyshev column, and the bed's admission | the safest-move-is-the-sink fraction at or above $0.5$ on the unrestricted bed; or the exact-zero-tie discard fraction at or above $0.5$ | $0$ GPU-s | V-12, V-8, V-10, V-5 |
| **R-08** no-op moves inflate the effective $m$ | every printed argmin floor | $m_{\rm eff}<m$ on more than 5 per cent of draws in a cell quoting a floor | $0$ GPU-s | L-FLOOR, V-17, V-10 |
| **R-24** a flag lookup decides the argmin | BED-S's argmin head | the closed-form rule at or above $\mathrm{acc}_{\rm shape}-\mathrm{MDE}_8$ on at least 6 of 8, or above the zero-information floor by more than its interval at construction | $0$ GPU-s | V-10, D-5, M-21 |
| **R-09** a bind has an empty rejection region | the proposition that bind carries | any plant passing the honest bar; Gaussian $V$ reading below $10^{-6}$; a non-causal $P$ reading zero displacement before the intervened row | seconds | V-24, V-14, V-3 |
| **R-12** the solve path is not deterministic or not priced as claimed | the "first gated wing under strict mode" sentence; every `[FITTED + RUN]` price | strict mode raises on the backward; or the cell reads above $2.2\times$ the law | $\approx1.7$ s | V-16, P-8, M-3 |
| **R-13** the capped seeds do not cross | the record's account of R1 | at most one of three crossing under the cap | $\approx9.65$ s | M-6, D-6 |
| **R-14 / K-9** the timer's strongest correlate is run order | every GPU-second in the lane | a Spearman correlation of seconds against run index comparable to the record's $+0.7029$; or any price quoted before the sweep | band $206$ to $537$ GPU-s | C17, P-8, M-10 |
| **R-15** a "matched" contrast is not matched | every "at matched parameters" sentence | a per-arm count differing by more than $0.032$ per cent after the head is attached | $0$ GPU-s | Ruling 3, M-8 |
| **R-16 / K-D2** the arm copied the environment | every capability number on BED-S | $\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at least 6 of 8 seeds | inside the pair | D-2, M-7 |
| **R-17** a guard partition postdates the itinerary it scores | every itinerary statistic | the partition's hash later than the first itinerary record | $0$ GPU-s | M-2, M-18 |
| **R-18 / K-P** the realised sd is too large for $N=8$ | the falsifiability of every $N=8$ kill | realised paired sd at or above $2.18\times$ the pilot ($0.109199$, giving $\mathrm{MDE}_8=0.126238$) | $\approx34$ s | M-3, M-9 |
| **R-19 / K-E1** a column sink reproduces the row condition | the boundary-row mechanism as a capability | ChaCAL-with-sink within $\mathrm{MDE}_8$ on at least 6 of 8 **and** residual within $2\times$, holding at $N=16$ | $\approx98$ s, escalation $\approx44$ s | V-24, M-13, M-7 |
| **R-20** the one-step rule is not a policy | the phrase "safest move" | one-step and two-step optima disagreeing on more than half of 512 oracle draws | $0$ GPU-s | P-7, V-17 |
| **R-21 / K-J** the dial is pinned | the horizon-dial sentence, Bet C, and S-64's runnability | PINNED on at least 6 of 8 under the boundary null; or the ablation moving NRMSE by less than one seed sd | $0$ GPU-s | Ruling 10-prime, M-20, V-9 |
| **R-22** parity by underpowering | any "parity" or "equivalent" sentence | such a sentence resting on $N=8$, where the interval half-width is $0.8807\sigma$ against a $0.5\sigma$ margin | $119$ to $228$ s | M-13, M-9, V-5 |
| **R-23** the skyline is unmatched or chosen late | Bet D's row | a depth chosen after the arena; or a "cannot" sentence outside the skyline table | $\approx7.6$ s per cell | P-10, R-SKY, K-10 |
| **R-25 / K-G1** the re-solve is not needed | component (g)'s re-solve sentence | the cached mixture within one seed sd of the shape on $P$-changing plants on at least 6 of 8 | $1.680$ s per cell | D-2, P-7 |
| **R-10 / K-I** the certificate is uninformative or wrong | every "certified" sentence | one exceedance on 1,024 draws; or $\delta\,\|V\|_\infty$ at or above the label sd | $\approx2.6$ s | L-CERT, V-10, V-17 |

*Withdrawn candidate, recorded so it is not re-found (V-7 discipline).* The conditioning of the committor solve on a self-loop-saturated transient block: with the diagonal logits raised by 0, 4, 8 and 12, $\max_T P_{ii}$ reads $0.6927$, $0.9919$, $0.99985$, $0.999997$ and the sup-norm condition number $7.9$, $214$, $771$, $810$, while the float32-against-float64 solve gap stays at $2.8\times10^{-8}$ to $3.3\times10^{-8}$ (`RUN[M]`). It does not bite at $s=32$; it survives as one printed line beside the committor head and is not a card.

---

## 5.7 The cost table by phase, and the Kaggle gate

**The basis.** Two fitted throughput laws on the certified card (`READ COSTS.md:73-74`): softmax at R-squared $0.9999975$ over five points, reading $0.010165$ s per step at $n=2048$; the corner-3 base at R-squared $0.99999987$ over three points (the two largest refused because they paged over PCIe), reading $0.093084$ s per step. The shape on the softmax corner is the softmax law **plus the solve increment** $1.041$ ms per step (`RUN[NEPTUNE]`: solve plus product $2.514$ ms against product alone $1.473$ ms, forward and backward, float32, $n=2048$), giving $0.011206$ s per step. Rules the prices obey: each arm at its own law (M-8); the solve is an **increment**, never a cross-arm ratio; the increment was measured at the shipped $s=64$ and never scaled from a smaller $s$ (M-3); it is a per-op floor under a $2.0\times$ to $6.6\times$ dispatch gap, so every shape price carries direction "at least" until the cell-in-band check runs (P-8); no price crosses the device boundary (V-22); the laptop clock is not stationary at plus or minus 12 per cent, so prices are quoted in GPU-minutes to two decimals.

| unit | price | class |
|---|---|---|
| one 150-step softmax control cell, $n=2048$, $s=64$ | $1.524$ s | `[FITTED]` |
| one 150-step shape cell on the softmax corner | $1.681$ s | `[FITTED + RUN]` |
| fixed cost per invocation | $4.0$ s | READ |
| one bed-cell pair, 8 seeds each | $29.6$ s in one invocation; **$34$ s** carried (two invocations, the conservative reading the record used) | DERIVED |
| the pair on the corner-3 base | $129$ to $133$ s, **no certificate on this base** | DERIVED |
| InfSA-style Neumann $K=16$ cell | $5.12$ s; $44.9$ s at 8 seeds | DERIVED floor |
| depth-5 skyline cell | $7.6$ s; $64.8$ s at 8 seeds | `[ASSUMED linear in depth]` |
| the capped cell | $1.884$ s | READ |
| 1,024 forward passes at $n=2048$ | $\le2.6$ s | DERIVED |
| paired equivalence test at $N=36$ / two-sample at $N=70$ | $119$ s / $228$ s | DERIVED |
| the 9,600-step ladder, one pair | $32$ to $36$ min | `[FITTED]` at 64 times |
| the $s$-sweep | $206$ to $537$ GPU-s, **band only, no point** | READ |
| Lean targets, manifests, censuses, verdicts from journals | $0$ GPU-s | — |

| phase | cards | GPU cost | class |
|---|---|---|---|
| **0** | J-L0 to J-L18, J-D1 to J-D8, S-01 to S-03, M-0.3, N-03, R-15, R-17, S-61, V-0 | $0$ | — |
| **1** | S-10 to S-17, R-06, R-07, R-08, R-24, R-20 | $\le16$ s ($0.27$ min) | DERIVED |
| **2** | M-2.0, S-20 to S-28, S-52, N-01, N-02, N-07, S-40, S-41, S-44, S-60 | $\approx107$ s ($1.8$ min); $+524$ s if the whole device certificate is rerun | DERIVED, N-02 `[ASSUMED]` |
| **3** | S-30 to S-35, S-62, S-51, S-50, M-3.7, M-3.8, V-17 | $\approx8.7$ to $9.2$ min | DERIVED, S-35 `[ASSUMED]` |
| **4** | S-63, S-64, S-65, R-21, M-4.2, S-42, M-4.5 | $\approx3.0$ to $4.8$ min | DERIVED |
| **local decision total, phases 0 to 4** | | **$\approx14$ to $16$ GPU-min** ($\approx25$ min with the full certificate rerun) | — |
| **5** | S-66, N-09, N-10, M-5.2, N-18 to N-21, M-5.3, N-11 to N-17, J-D5, S-72, J-D8 | $\approx35$ to $45$ min for the priced rows; the kernels are `NOT MEASURED` | READ band plus `[FITTED]` at 64 times |
| **local total, phases 0 to 5** | | **$\approx0.8$ to $1.0$ GPU-h** | — |
| **6** | N-22, N-23, N-24, S-73 | $0$ locally; on Kaggle $\approx8.7$ GPU-min `[ASSUMED]` for the certificate, the LM cell `NOT MEASURED` | behind the yes |

The whole falsification programme to a verdict costs under a quarter of a GPU-hour on the certified box. The record's dominant failure — a bed dying at construction — is a $0$ GPU-s decision at S-12, and the expensive death (K-E1) is a $\approx34$ s reading escalated once for $\approx44$ s.

**What fits, and what does not.** Bytes are modelled as
$$\text{bytes}\;\approx\;1.256\Big[C_{\rm OP}\,n\,s^2\cdot4+2\,n\,s\,d\cdot4+17.874\,n\,s\,d\cdot4\Big],$$
with $1.256$ the worst reserved-over-allocated ratio, the residual constant $17.874$ at R-squared $0.996373$, and $C_{\rm OP}\in[3.823,5.823]$ — the signed arm's fitted value plus one to two retained tensors, `[ASSUMED]` until N-18 measures it — against a budget of $0.90\times7.996=7.20$ GiB. At the carried upper constant the largest power-of-two $n$ per $s$ is $32768$ at $s=64$, $2048$ at $s=256$, $128$ at $s=1024$ and $8$ at $s=4096$; the dense operator at $n=2048$, $s=4096$ is $137$ GB and runs nowhere. So $n$ is declared per $s$ on every long-context card, and the corner-3 base, at 8 bytes per element, halves every $n$ and is not swept.

**The Kaggle gate.** Nothing in this plan launches. The gate's local half (N-24) is: the device certificate lines filled by N-01, N-02 and N-18; the envelope's two plants firing (N-22); the package's smoke passing (N-23); the pinned bed digests re-asserting on regeneration (Ruling 7); the corpus pin with its split; the three unpinned sources whose hash cell prints and then **raises**; and the open rulings listed by name beside the request. The gate's own circularity — its first rows are blocked on numbers only the Kaggle run produces — is the author's ruling and is not taken here. Its only consumer in this programme is S-73, the LM $\gamma$ cell. **The rule, from the author's own side and this plan's standing memory: notify first, and no launch, upload or training start without an explicit yes.** A threshold carried across the device boundary is V-22 and is refused; a Kaggle number pooled with a local one is struck.

---

## 5.8 The Lean ledger

Grades: `[M]` a Mathlib route is confirmed at the pinned revision and the statement is expected to elaborate; `[S]` a supporting lemma must be supplied in-file; `[D]` deferred, no object exists in this Mathlib. **Every `[M]` is pending `lake build`; nothing is cited as proved before it builds** (L-LEAN, P-11). The eight Mathlib names the `[M]` rows rest on were found by `grep` in the vendored tree at rev `a45ae637` this session; a card introducing a further name is `[S]` until that name is checked.

| # | target | grade | depends on | what it licenses | what dies if it fails |
|---|---|---|---|---|---|
| 1 | `gamma_zero_is_softmax` with `gamma_half_is_not_softmax` | `[M]` | J-L0 | "CEQ contains softmax at $\gamma=0$ by theorem" (the equation is ChaCAL's; the machine check is the record's) | the Lean rejection region of parity; Proposition 1 reads RUN-only ($2.3003$) |
| 2 | `bos_row_is_absorbing` at $\beta=1$, with the $\beta=0$ refusal | `[M]` | J-L0 | census line X-2 as a theorem instance; J-L10 may take "BOS declared" as its only hypothesis | BED-S admission needs the determinant check on every batch instead |
| 3 | `later_boundary_unreachable` | `[M]` | J-L3 | F2 as a theorem: a set after the query has committor exactly zero | census line X-4 stays a RUN with its plant |
| 4 | `softmax_corner_not_nilpotent` | `[M]` | J-L0 | "all $s$ hops by nilpotency" is a regime-N sentence only; the default keeps the diagonal | nothing measurable; the RUN pair ($6.27\times10^{-13}$ against $7.96\times10^{-8}$) stands |
| 5 | `lower_triangular_isUnit`, `diag_one_sub_smul_pos` | `[M]` | J-L0 | Propositions 5(a) and 10; every card below that inverts a triangular matrix | the triangular branch stalls one evening; the substitution stands as RUN ($1.8\times10^{-15}$) |
| 6 | `resolvent_fromBlocks`, `segmentation_blockdiag` | `[M]` | J-L0 | the contract's F0 line finally has a declaration; certificate Z-F0 at $\delta=0$ is a theorem instance | the F0 row keeps its `torch.equal` RUN and no `[M]` tag |
| 7 | `cut_makes_segment_head_absorbing`, `cut_severs_boundary_sets` | `[M]` | J-L6, J-L2 | the dial rule that every segment carries its own sink, goal and constraint sets | segmentation and the committor channel are kept apart by rule, not by theorem |
| 8 | `displacement_identity` with `const_value_zero_displacement` | `[M]` | J-L0 | Proposition 7(i) as LEAN, cited beside the linearisation and the rank-one form | nothing; three independent RUNs stand at $10^{-15}$ |
| 9 | `lowerTriangular_ne_symmSupport` | `[M]` | J-L0 | the DAG substrate ruling as a theorem consequence: no causal operator equals an undirected chain | the substrate ruling stands on the RUN alone |
| 10 | `committor_is_resolvent_read` (a), (b-causal) | `[M]` | J-L3, J-L2 | Proposition 10: "absorption almost surely" is the single census line "BOS declared" | the head ships on a per-batch determinant check |
| 11 | `subdiag_pow_entry`, `pathprod_is_chain_resolvent` | `[S]` | J-L0 | Proposition 2 in both halves; BED-M is **contained by theorem** and `shape - corner-3` is VOID by theorem | the containment stays a $0.0$ RUN, VOID by registration |
| 12 | `neumann_truncation_bound`, `neumann_tail_attained`, `mask_amplification` | `[S]` | J-L0 | Proposition 4(i) and (ii); certificate rows Z-NEU and Z-F1 | every $\delta$ is printed from the textbook with the RUN plant; no arena cell is lost |
| 13 | `resolvent_is_triangular_solve`, `mixing_matrix_rowStochastic` | `[S]` | J-L3, J-L12 | Proposition 5(a) to (c); the hull bound applies verbatim; the cost line prices a substitution that exists | the cost law stands on the RUN |
| 14 | `causal_forward_only`, `suffix_resolve_eq_full`, `sherman_morrison_row` | `[S]` | J-L4, J-L8 | Proposition 7(ii) and (iii); the safest-move price is a theorem about the **oracle's** clamp, never carried to a token rewrite | the price is a DERIVED band with the RUN identities |
| 15 | `isUnit_one_sub_of_perron`, `reach_avoid_sum_one` | `[S]` | J-L10 | Proposition 8's conservation row and degeneracy lemma; the goal set and declared sink are forced by theorem | the row is printed with its residual per batch and the lemma is DERIVED |
| 16 | `masked_sink_finite_sum` | `[S]` | J-L12, J-L10 | "BOS is a value-zero sink set" as a theorem, not a convention | the paper keeps the diagonal and Proposition 4 everywhere, already the default |
| 17 | `hitting_time_transform`, `f1_cantelli_union` | `[D]` | J-L0 | nothing yet; the content is carried as J-D3 and J-D7 | nothing; the statements live outside the root build |

Rows 1 to 10 are one evening each of independent `[M]` work; rows 11 to 16 name their dependencies; row 17 is off the critical path. Price: $0$ GPU-s each. The gate that consumes them all is J-L18, and its census column is what turns a green build into a licence: a theorem whose hypotheses no draw of a bed satisfies is **decoration on that bed** (L-DOM), which is why `segmentation_blockdiag` reads `3/3` on BED-M and `0/N` on BED-S's softmax corner and the paper says so in both places.

---

## 5.9 The leap-dossier discipline

The author's supersession of 2026-09-03 licensed a new primitive; it did not lift the leap's other clauses. This is what a single leap call would receive, and the rule that governs what may be done with what comes back.

**What the call receives.** (i) The shape's definition and the eleven propositions with their status tokens and their `MISTAKES.md` mechanisms, as `judge/sec_shape.md` states them. (ii) The Lean ledger of §5.8 with each row's grade and each row's refusal. (iii) The apparatus: beds with the F1 and F2 conventions, labels and heads, floors, skylines and fellow approximators, control arms, metrics, census lines, binds with their planted negatives, certificates in vector units, the identity manifest field list, and the pricing rule. (iv) The predictions ledger of §5.5 with counters and SPLIT bands, and the calibration column (7 of 8 optimistic, $p=0.0352$). (v) The standing attacks of §5.6 with frozen thresholds. (vi) The `NOT FOUND` list and the eight prior-art sweeps' queries, so absence is bounded by what was searched (V-7). (vii) The named fields the record's own leap ledger already owes — realisation theory, Koopman spectral theory, bifurcation and gradient-flow convergence, structured sparsity scheduling, finite-mixture inference, anti-concentration, error-statistical severity, implicit bias — with the field-naming rule: naming a field and then saying what is wanted from it is compliant, naming the want and calling it a field is not.

**The rule.** A leap output is filed `[LEAP-UNTESTED]` with, per claim, its counter of equal specificity and its cheapest killer. **Its instances run before its text is believed**, and a leap output acted on before its instance runs is **struck** — that is the R15 kill, unchanged by the supersession (`READ CEQ_V20_R15_CONTRACT.md:265-273`). A second leap call in the same round is a breach. An absorbed component without a licensing theorem is dropped. The leap is scored in the same calibration column as the author, in its own sub-column, on instances-run and predictions-falsified — never on how interesting the text was.

**What a dossier row looks like when it is complete** (M-4.5): the candidate built with identity binds that each failed their plants at $O(1)$; the manifest with every declared field present; the certificates in vector units with $\|V\|_\infty$ and $1/(1-\hat\gamma)$ printed; the parameter count per arm; the verdict by the clauses registered before the run; the purge, with the manifest committed
**before** any deletion; and the prognosis naming one arm, its dossier, its cost, its certificate, its floors and its one sentence. A bracket filled from prose rather than from a `results/` row voids the row (P-1, P-2).

---

## 5.10 What dies if what

Three outcome trees. Each names the sentence it licenses and the replacement route it owes. Under D-CALIB-1 the middle tree is the planning estimate.

**Tree A — every prediction holds** (the half the calibration column says is less likely). J-L18 reads `10/10`; S-12 admits; S-26 strikes nothing; the arena reads the arm identified in $[10^{-3},10^{-2})$, $\hat\gamma$ MOVED on at least 6 of 8, the residual ratio at least $10$, the cosine above softmax's by more than $\mathrm{MDE}_8$, the argmin interval below the exact floor; S-31 and S-64 separate the shape from ChaCAL-with-sink; S-35 lands inside $\mathrm{MDE}_8$; V-17 reads at or below $0.5\times$; S-65 declares equivalence at $N=36$. *Sentence licensed:* "Consequence-Equilibrium Attention is ChaCAL's causal resolvent read on the record's Lean-checked three-corner base with a declared sink, a goal set and $K$ constraint sets as boundary rows; on BED-S at $t^\star=8$, $m=8$, $K=2$, $N=8$, parameters recounted per arm, it reads the reach-avoid vector with argmin error $[e]$ (interval $[l,u]$) below the exact restricted-view floor $[f]$, harmonic residual ratio $[r]\ge10$ at equal marginal error, and displacement cosine $[c]$ above depth-1 softmax's by $[\Delta]>\mathrm{MDE}_8$; ChaCAL at the same $\hat\gamma=[g]$ ($\Lambda=[\Lambda]$, MOVED) with a sink token reads above the floor; the depth-5 skyline reaches the same argmin accuracy within $\mathrm{MDE}_8$, so the separate advantages are exactness ($\delta=0$ on the exact route), one-read cost (one solve, depth $s$, plus 50 per cent causal MACs) and the boundary-row mechanism." Every bracket is a number BED-S has not produced. *Route owed:* none; the ladder (M-5.3) is spent and the price is re-quoted upward.

**Tree B — the median outcome** (every counter is the point estimate). J-L18 reads `9/10`, one row demoted to `[S]` on an unconfirmed Mathlib name. S-12 admits after one repair of the sink share — the fix's own risk, since the declared sink read $[0.362,1.000]$ on the one draw run. S-26 does not strike. The arena reads the arm identified, $\hat\gamma$ PINNED on at least 6 of 8, the residual ratio SPLIT in $(2,10)$, the cosine within $\mathrm{MDE}_8$, the argmin interval straddling the floor. S-31 and S-64 are then void, because ChaCAL at $\hat\gamma=0$ **is** softmax. S-35 lands inside $\mathrm{MDE}_8$; V-17 is SPLIT; S-65 needs $N=70$. *Sentence licensed:* "Consequence-Equilibrium Attention is a re-parameterisation of ChaCAL's causal resolvent read with absorbing boundary rows on the record's three-corner base. Nine of its identities are machine-checked and the tenth is `[S]` with its numeric instance; the committor head is the exact solve at $\gamma=1$, matching the bed's own Dirichlet solve to $0.0$ on its real sets; BED-S is registered with an exact oracle at $0.0$, the exact zero-information floor $0.875$ at $m=8$, and a printed paired sd $[\mathrm{sd}]$. At $N=8$ the trained $\hat\gamma$ is not distinguishable from zero under the boundary null, depth-1 softmax matches the marginal error and the displacement cosine within $\mathrm{MDE}_8$, and the residual ratio is SPLIT. No capability sentence is licensed; the contributions are the identities, the certificate discipline, the bed with its floors, the controls and the negatives." *Routes owed:* **retire** the horizon-dial sentence to a definition (R-21) and the LM $\gamma$ claim (S-73) before it is made; **reprice** "one read" as a cost statement (Bet E) and the argmin to $N=16$ (about $68$ s) before any argmin sentence; **retire** "consequence as a field" to a label class (Bet A); **keep** regime N at $\gamma=1$ (corner 3, exact without a dial) and the constant-value plant. This is the sentence the author reads first.

**Tree C — the shape dies at its weakest component, before a GPU-second.** BED-S does not admit at the design point: the sink share dominates, or the admitted-region sd of the goal committor reads at or below $0.05$, or an argmin class frequency leaves $(0.05,0.95)$, or R-07's Chebyshev ties empty the bed, or R-24's flag rule reads above the floor at construction. Everything from S-14 to S-65 is VOID and unscored. *Sentence licensed:* "BED-S at the design point does not admit; the paper files the generator specification, the census with its printed failure line, the floors, the controls and the pricing rule, and no capability number." *Routes owed, in order:* **reroute the dial** — move $t^\star$ to 2 or raise the graph depth, since the DAG substrate bounds $t^\star$ by the causal window, and re-run the $0$ GPU-s census; **reroute the substrate** — the jittered `bed_1` landscape at $K=2$ with $B$ as goal, as an **oracle cross-check only** (it is undirected with $\rho(Q)=0.9409$, and no causal operator equals it, so it is never the shape lane's substrate); **reprice $N$** from whatever realised sd the first admitted batch shows — at the $2.18\times$ precedent, $\mathrm{MDE}_8=0.126238$ and nothing in Phase 2 is falsifiable at $N=8$, so $N=16$ is the first honest count.

**The smaller deaths, one line each, with what survives.**

| kill fires at | what dies | what survives, and at what price |
|---|---|---|
| J-L18 red (K-K) | the "machine-checked" sentence for the rows that failed; training does not start | every float instance as RUN; the arena waits one evening |
| S-12 census ($0$ GPU-s) | BED-S at $t^\star=8$ | the reroute above, repriced from its own sd |
| S-26 / R-06 (K-H2, $\approx0.5$ GPU-min) | BED-S as a capability bed | the label class registered with printed floors; no number |
| R-07 ($0$ GPU-s) | the Chebyshev column as written | the goal rule and the conditioned column, with the restriction in the spec |
| R-24 ($0$ GPU-s) | BED-S's argmin head | the committor-vector head alone |
| S-20 (about $1.7$ s) | the determinism sentence, or every `[FITTED + RUN]` price | the corner-3 base at $133$ s per pair, with no certificate, and the paper says why |
| S-62 K-D2 ($\approx34$ s) | every capability number on BED-S | the identification reading; the property paper |
| R-21 PINNED ($0$ GPU-s) | the equilibrium sentence; S-64 and S-65 become void | $\gamma$ as a bed-side dial; regime N at $\gamma=1$, exact without a dial |
| S-64 K-E1 ($\approx44$ s) | component (e) as a mechanism | ChaCAL plus a certificate plus Lean containment plus a registered bed — tree B's sentence |
| S-33 K-G1 ($\approx35$ s) | component (g)'s re-solve sentence | the displacement identity as algebra; the channel as a cost statement |
| Bet E at or below 2 | "joint determination in one read" | the residual as a diagnostic; one solve, depth $s$, as a cost statement |
| S-41 / N-15 K-I ($\approx2.6$ s) | the mask | the exact solve, cheaper than one hop at $s=64$ |
| S-66 K-9 (band) | every $s$-scaled price and the "plus 50 per cent" wall-clock sentence | the MAC law as a MAC law; measured points quoted as points |
| N-24 (the yes withheld) | nothing local | the whole local programme; the Kaggle rows stay `NOT MEASURED` |

---

## 5.11 Deliverables, the return report, and how to keep this plan current

**Deliverables of the paper.** `docs/CEQ_SHAPE.md`, `docs/CEQ_SHAPE.tex`, `docs/CEQ_SHAPE.pdf`, `docs/references.bib`, and `docs/PLAN.md` — this section, standalone.

**Deliverables of the programme, by phase.** Phase 0: `lean/CEQ/Shape/*.lean` with `docs/LEAN_SHAPE_MANIFEST.md` and `docs/LEAN_SHAPE_CENSUS.md`; `docs/apparatus/MANIFEST_SHAPE_LANE.md`, `IDENTITY_SCRIPT_RULES.md`, `OBSTRUCTION_CENSUS.md`, `INSTRUMENT_GAMMA_LR.md`; `docs/PRICE_LEDGER.md`; `docs/CEQ_KERNEL_PATH_BATTERY.md`; `docs/CALIBRATION.md`. Phase 1: `docs/beds/BED_S_PLACEMENT.md`, `BED_S_SPEC.md`, `BED_S_CENSUS.md`, `BED_S_VOID_LIST.md`; `docs/apparatus/METRICS_AND_FLOORS.md`; `results/bed_s_census.jsonl`, `bed_s_zerohop.jsonl`, `bed_s_leak.jsonl`, `bed_s_dz_census.jsonl`. Phase 2: `results/binds/*.jsonl` with one FOUND cell per plant; `docs/apparatus/PLANT_REGISTER.md`, `CERT_EXACT_ROUTE.md`, `CERT_F0_SEGMENTATION.md`; `results/certs/z_neu.jsonl`; `results/bedm/capped_seeds_2_3_7.jsonl`; `results/k_cert_local.json` with its two new blocks; `docs/apparatus/PREDICTIONS_R16.md` and `results/arena/PREDICTIONS.sha256`. Phase 3: `results/arena/*.jsonl` per arm, `results/bedk/first_cell.jsonl`, `results/bedm/{containment,vector_lane}.jsonl`. Phase 4: `results/arena/VERDICTS.jsonl`, `docs/apparatus/ADJUDICATOR_SPEC.md`, `docs/plan/GAMMA_VERDICT.md`, `docs/CEQ_SHAPE_DOSSIER.md`. Phase 5: `docs/COST_LAW.md`, `results/cost/*.jsonl`, `docs/CEQ_{CHUNKED_SOLVE,CSR_PATH,MAPPER_SCHEDULE}.md`, `docs/apparatus/DIGRAPH_BETA0_SPEC.md`, `PERRON_WEIGHT.md`. Phase 6: `docs/CEQ_ENVELOPE.md`, `docs/KAGGLE_GATE.md`, `MODEL_CARD.md`.

**The report the author finds on return.** One file, `docs/PLAN_STATUS.md`, regenerated at the end of every working session and never edited in place, with exactly these blocks and no prose beyond them:

1. **Where the DAG stands** — every card id with a token in `{not started, in progress, PASS, KILL, VOID, NOT MEASURED}` and, for a PASS or KILL, the deciding number with its evidence class.
2. **What was decided since the last report** — one line per gate crossed, naming the gate, the numbers that decided it and the phase it opened.
3. **What died, and what it took with it** — the row of §5.10's table that fired, quoted, with the sentence now withdrawn and the replacement route owed.
4. **The GPU-minutes spent** against the phase's budget in §5.7, with the class of each price.
5. **The calibration column** — rows scored, rows wrong, signs, and the running one-sided sign test on the pooled column and on the author and leap sub-columns separately.
6. **The next five cards**, cheapest-decisive first, each with its price and its evenings.
7. **Open questions for the author only** — the Kaggle yes, and any ruling the plan cannot take alone. Nothing else in the report asks a question; every other line is a decided number.

**How to keep this plan current.** An **append-only corrections index**, `docs/PLAN_CORRECTIONS.md`, one row per correction: `id, date, what was wrong, the number or line that says so, the card or section corrected, supersedes`. The plan's body is never silently edited: a corrected card gets a new row and a `supersedes` pointer, exactly as journals do (L-G2), and the round-15 corrections index C1 to C40 is the pattern. The mechanism is **P-3** (a stale claim never retracted): the record's own instance is a prompt calling a direction "the one live direction that has not yet been measured here" while a complete 305-line instrument for it sat in the tree with two of six required cells already journalled, and a document saying a run "was still running when this was written" that finished and recorded a worse number and was never updated (`READ MISTAKES.md:316-328`). The rule the correction index enforces is that one: **a claim of the form "not yet measured" or "still running" carries the date it was written and is checked against `results/` before it is read.** Three standing consequences for this plan: every `NOT MEASURED` tag carries the instrument it needs and is re-checked at each report; every `[ASSUMED]` price carries the card that would replace it; and every `[U]` citation carries what would make it `[V]`. The debts open at the time of writing, so that they are not rediscovered as findings: the boundary-null critical value $2.7055$ has no `references.bib` citation and is **owed**; ChaCAL's diagonal convention rests on one HTML fetch and must be re-read against the PDF before S-31 is typeset; the adjacency-feature representability of the environment chain inside the arm's class is unmeasured; the Kirchhoff second route covers $K=2$ single-node sets only; the hop-to-committor reduction is NOT FOUND; and BED-S has no cell, no realised sd and no measured $t^\star$, so every BED-S number in this plan is a floor formula or a design constant.
