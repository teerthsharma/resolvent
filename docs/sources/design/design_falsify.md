# DESIGN — the shape as a set of bets (falsification-first)

*VENUS / IRENE, 2026-09-03, HEAD `207e7b9`. Written without reading the other two design files.
Evidence classes: `RUN` (executed this session, this box, torch 2.5.1 float64 CPU unless stated),
`READ path:line`, `CITED [V]` by BibTeX key in `$SCRATCH/sweep/bib_*.bib`, `DERIVED` (steps
shown), `ASSUMED` (labelled, never load-bearing). Every bet names the `MISTAKES.md` mechanism it is
designed against. Prices are on the certified RTX 4060 Laptop (`COSTS.md:53-60`, READ via
`sec_cost.md` §4.x.8), quoted to the ±12 % the record allows (`V17_R4_RETAKE_PRICE.md:143-150`).*

---

## 0. The rules this document is written under

**0.1 What a bet is.** A prediction with a number, a counter-prediction of equal specificity
(L-SIGN, `CEQ_V16_CONTRACT.md:58-61`, READ via `sec_state.md` S.4), the cheapest run that
separates them, the number that fires the kill, its price, and the one-sentence claim the paper
is left with under each outcome. A prediction without its counter is D-7 (`MISTAKES.md:2037`)
and is not filed here.

**0.2 The calibration discount, applied as the record wrote it.** The record's only scored
prediction census reads 9 checked, 9 adverse, 7 of 8 signed rows optimistic, one-sided sign test
`p = 0.0352` (`V16_CALIBRATION.md:17-19, 96-100`, READ via `sec_measured.md` M.5); Wilson 95 % on
the optimism fraction `[0.5291, 0.9776]` excludes `0.5` and fixes nothing else (`:171-184`). The
rules are the record's own (`V16_CALIBRATION.md:142-203`):

- **D-CALIB-1** — the counter is the point estimate. Every "median outcome" below is the counter.
- **D-CALIB-2** — a bare prediction is blocked, not discounted.
- **D-CALIB-3** — sign, never size: no numeric shrink factor is applied anywhere.
- **D-CALIB-4** — the cheapest refutation of the optimistic half runs first (§5's order).
- **D-CALIB-5** — the row is appended whether or not it flatters.

The ranking in §5 is therefore ordinal, driven by evidence the record already holds against each
component; any probability word in it is a `GUESS` and labelled.

**0.3 What the sweep did to the bets before they were placed.** The read operator is
`fagnou-2024-chacal` [V] up to the scalar `(1−γ)` (`sweep_resolvent.md` §2.1; `sweep_occupied.md`
§0.1); the absorbing-chain *reading* is `roffo-2026-infsa` [V]; the resolvent as a layer is
`gasteiger-2019-appnp` [V]; the clamped-boundary committor as a learning layer is
`zhu-2003-harmonic` [V]; the barrier re-solve on a fixed chain is `piray-2021-linearrl` [V] Eq. 5;
the causally-masked fixed point with `do()` by clamp-and-re-solve is `scetbon-2024-fip` [V].
Nothing below bets on any of those. The bets are on the eleven NOT-FOUND components — the eight of
`THESIS_CORRECTIONS.md` C1 plus three the sweeps file separately (§3.9–3.11) — and on the three
capabilities the author named.

**0.4 The obstruction-4 correction is binding.** For row-stochastic `P` the mixing matrix
`(1−γ)P(I−γP)^{-1}` is non-negative and row-stochastic (`THESIS_NOTES.md` P5(b);
`sweep_resolvent.md` §5.2), so absorption *redirects* mass and never subtracts it (C3). No bet
claims a signed influence; the `ceq/attention.py` min-entry test is *predicted* to read `0`.

---

## 1. The price list every bet draws on

`[FITTED]` from `results/k_cert_local.json` laws or `[RUN]` per-op microbenchmarks as
`sec_cost.md` §4.x.8 assembles them; the per-op solve increment is a **floor** on the end-to-end
increment (P-8, `MISTAKES.md:387`) because dispatch is not in it (`scale/m3_flops.py:101-121`:
`2.0×–6.6×`).

| item | price | class | source |
|---|---|---|---|
| one 150-step cell, shape on the softmax corner, `n = 2048, s = 64, d_model = 16`, 4,769 params | `1.680 s` (law `0.010162` + solve `0.001041` per step) | FITTED + RUN | `sec_cost.md` §4.x.8 |
| the softmax control cell | `1.524 s` | FITTED | same |
| **one bed-cell pair, 8 seeds each, + 4.0 s fixed** | **`≈ 34 s`** (`17.4 + 16.2`) | FITTED + RUN | same |
| the pair on the corner-3 base (`arm_smprime`, complex64 path product) | `≈ 133 s` | FITTED + RUN | same |
| full ladder, 9,600 steps, one pair | `≈ 36 min` | FITTED ×64 | `V17_R4:243-254` |
| ChaCAL control (same operator, no boundary rows) | `1.680 s` per cell | DERIVED (identical arithmetic) | `sweep_resolvent.md` §2.1 |
| InfSA-style Neumann read, `K = 16`, per step at `n = 2048` | `25.409 ms` vs solve `2.514 ms`; cell `≈ 5.1 s` | RUN → DERIVED floor | `sec_cost.md` §4.x.3 |
| depth-5 softmax skyline (`⌊log₂ 8⌋ + 2`), unmatched params | **NOT MEASURED**; `≈ 7.6 s` per cell | ASSUMED (linear in depth) | `sec_beds.md` §6.C.6 |
| BED-S oracle, `m·K` solves at `|T| = 1200` | `≈ 5.8e8` flop per solve, `< 1 s` per draw, once | DERIVED | `sec_beds.md` §6.D.5 |
| identity checks (I1–I5, C6, EMC, planted zero) | seconds, CPU float64, 0 GPU-s | RUN | §4 |
| TOST parity at power 0.80, `N = 70`, one pair | `≈ 70 × 3.204 + 4 ≈ 228 s` | DERIVED | `sec_beds.md` §6.D.3 |
| one Lean target of the `V15Source.lean` scale (201 lines, 7 declarations) | 0 GPU-s; one file | READ `sec_proved.md` §2.8 | — |
| MDE at `N = 8`, paired sd `0.034451` | `0.039827` | RUN (`sec_beds.md` §6.D.2) | `MATHEMATICS.md:979-993` |
| zero-hop Fano floor on the argmin, `m = 4 / 8 / 16` | `0.5 / 0.6667 / 0.75` | RUN this session | `sec_beds.md` §6.C.5 |

Determinism prices reproducibility: `solve_triangular` is bitwise forward and backward over 8
repeats under strict mode on this card (RUN, `sec_cost.md` §4.x.6) while `cumprod`'s backward
raises (`COSTS.md:151-154`); the shape on the softmax corner is the first arm in the record whose
*training* step runs under `use_deterministic_algorithms(True)`; the corner-3 base cannot.

---

## 2. The three capability bets (the author's message 2)

Each is placed on BED-S at `sec_beds.md` §6.E's design point — `t* = 8, m = 8, K = 2` plus a goal
set `𝒜_0` (C5), `N = 8`, 4,769 parameters, 150 steps, one thread lane, softmax corner as base —
and is **void** if any guard of `sec_beds.md` §6.C.4 fails at construction (D-4).

### 2.1 Bet A — consequence displacement (capability (a))

**Object.** `Δz(a) = z(do a) − z ∈ ℝ^{s×d}`, exact from the environment chain by
`Δz = (I − γP')^{-1}(ΔV + γΔP z)` (C6; **RUN this session**: `1.36e-15` on a causal softmax `P`,
`s = 64, d = 16, γ = 0.5`, row 20 redirected to `e_5`; `Δz` is exactly `0.0` on every row before
the intervened position and `1.127` max-abs after; the planted `V = 1` gives `Δz ≡ 0.0` exactly
while random `V` gives `1.127` — the bind has a rejection region, V-24). Scored by the
position-matched per-coordinate NRMSE vector plus field cosine and magnitude ratio
(`sec_beds.md` §6.B.3), paired against depth-1 softmax on byte-identical draws by McNemar
(`MATHEMATICS.md:389-395`).

**Prediction.** The shape's field cosine `⟨Δẑ, Δz⟩/(‖Δẑ‖‖Δz‖)` exceeds depth-1 softmax's by more
than the `N = 8` MDE of the realised paired sd (`0.039827` at the record's `0.034451`), and the
sign task at one position shows *no* separation: softmax already reads `0.807843` CP
`[0.754044, 0.854329]` there (`MATHEMATICS.md:396-406`) and the shape is predicted inside it.

**Counter.** The field cosines differ by less than the MDE: a per-row mixture trained on the
vector label learns the displacement coordinate-by-coordinate as well as the joint read does
(D-1 in vector form; `zhang-2023-cina` [V] shows single-effect estimation is softmax's ground,
`sweep_causality.md` §2.D). Then "consequence as a field" is a label class, not a capability.

**Cheapest killer.** One BED-S pair with the `Δz` head, `≈ 34 s`; fires at
`cos_shape − cos_softmax ≤ MDE(n = 8)`. Before it (D-CALIB-4), the free **zero-hop control** — a
per-position MLP on the intervened token's own row; within the MDE of the shape ⇒ the label is
static (`sec_refuted.md` C8) and the bed is struck before a number is quoted.

**Sentences.** Holds: *"On BED-S the displacement field under `do(a)` is read jointly by one
resolvent solve with field cosine `[x]` against depth-1 softmax's `[y]`, McNemar `p = [p]` at
`N = 8`; the sign of any one coordinate is read equally well by both."* Counter: *"The displacement
field is a vector label on which depth-1 softmax and the shape are indistinguishable at `N = 8`;
the consequence channel is exact by construction (C6, `1.4e-15`) and carries no capability
claim."* Zero-hop fires: the bed is struck.

**Mechanisms.** D-2 (oracle on `P_a`, arm sees edge tokens; `sec_beds.md` §6.C.4 guard 3); V-24
(`V = 1` plant, RUN); D-5 (`Δz` credited only where the oracle moved, `MISTAKES.md:755`); V-26
(cosine is joint); D-1 (the sign column is kept *because* softmax owns it).

### 2.2 Bet B — the safest move under `K` constraints (capability (b))

**Object.** Label `(q^{(0)}, …, q^{(K)})(a)` per candidate move with `Σ = 1` (**RUN this session**
on a two-set chain: row sums `[0.99999999999999978, 1.0]`, `min_i max_k q_k = 0.612 ≥ 1/K = 0.5` —
the C5 degeneracy is real and the goal set is mandatory); `a* = argmax_a q^{(0)}(do a)` with the
Chebyshev `argmin_a max_{k≥1}` printed beside it (`vanmoffaert-2013-chebyshev`,
`summers-2010-reachavoid`, `hsu-2023-safetyfilter`, all [V]). Two heads: the `[m, K+1]` committor
tensor in the Fisher–Rao coordinate `2 arcsin √p` declared before the run (M-2), and the argmin
(accuracy, Clopper–Pearson).

**Prediction.** `CP_upper(err_shape) < Fano_floor(k = 1)` at `N = 8`, and committor NRMSE below the
one-hop budget ceiling `NRMSE(z_1, z*)` (`sec_beds.md` §6.E P1, carried unchanged).

**Counter.** `CP_lower(err_shape) ≥ Fano_floor(k = 1)`: the joint read is no better than a one-hop
window licenses. At `m = 8` the zero-hop floor is `0.6667` (RUN); a shape at chance (`0.875`)
refutes by `+0.208`.

**Cheapest killer.** One pair, `≈ 34 s`, after the construction census prints label sd `> 0`,
argmin class balance, goal reachability and the discard count (V-8, V-12, V-25, D-3). Fires at
`CP_upper(err_shape) − Fano_floor(k=1) ≥ 0`.

**The second bet inside this one (C2).** ChaCAL with the same `γ` and a *sink token* standing in
for the boundary rows is the planted negative for component (e). Prediction: its committor NRMSE
exceeds the shape's by more than the MDE. Counter: within the TOST margin at `N = 70` (`≈ 228 s`),
in which case boundary rows are a parameterisation of ChaCAL's sink and (e) is not a capability.
The `N = 8` reading (`≈ 34 s`) runs first and is reported as "not separated at `N = 8`", never as
parity (M-13).

**Sentences.** Holds and ChaCAL-with-sink fails: *"On BED-S (`t* = 8, m = 8, K = 2` + goal) the
shape's argmin error `[e]` (CP `[l, u]`) sits `[Δ]` below the one-hop Fano floor `[f]`; ChaCAL with
the same `γ` and a sink token reads `[e']`, above the floor; the boundary rows are the mechanism."*
Holds but ChaCAL-with-sink matches: *"The reach-avoid vector is readable in one resolvent solve;
ChaCAL with a sink token reads it equally well; boundary rows are a parameterisation, not a
capability."* Counter: *"BED-S's safest move is not read better than a one-hop window licenses at
`N = 8`; the label class stands as a registered bed with printed floors."*

**Mechanisms.** V-8 / V-12 (goal set, sd print, discards); V-10 / D-5 (query token carries `s₀`
only, `I(s₀; a*) = 0`, untrained arm at `1/m ± CP`); D-2 (leak guard firing both ways,
`PASS_BAR = 0.5` / `FAIL_BAR = 0.9`, `scale/rips_gate.py:60-61`); M-2 (rule and coordinate fixed);
V-24 (ChaCAL + same `γ`); M-13 (parity only at `N = 70`); `misra-2023-safety-constrained-mdp` [V]
(Bellman optimality can fail on multichain CMDPs) goes to Limits.

### 2.3 Bet C — next transient state / phase / equilibrium (capability (c))

**Object.** The horizon dial: `γ = 0` reads the next step (softmax, bitwise, I1); `0 < γ < 1` reads
`E[γ^{τ−1}]` (`THESIS_NOTES.md` P3; the discounted safety value of `fisac-2019-bridging` [V]);
`γ ↑ 1` reads basin membership `argmax_k q^{(k)}`. "Phase" is the metastable-state membership of
`prinz-2011-markov` [V], not a new object.

**Prediction.** With the hop-ladder budget ceilings printed at construction
(`e4_harmonic.hop_reading`, `:194-199`), the shape's committor NRMSE at trained `γ̂` sits below the
ceiling at `k = 4` of `t* = 8` hops — half the budget in one solve — and `γ̂` is MOVED under the LR
test (`Λ > ln n`, Ruling 10′, `V17K_RULINGS.md:389-436`).

**Counter.** `γ̂` PINNED (`Λ ≤ 3.841`) on `≥ 6/8` seeds — softmax wearing a name (`sec_refuted.md`
C5; precedent: the hop-2 gain sweep `γ ∈ {0, .05, .10, .25, .50, 1.0}` where no `γ` beat `0`,
`workdonenew.md:276`) — or NRMSE above the `k = 1` ceiling, in which case the dial reads one step
and the equilibrium sentence is withdrawn.

**Cheapest killer.** The same `≈ 34 s` pair reading `γ̂` per seed with its LR statistic, plus the
ablation `(I − γ̂P̂)^{-1} → I` at trained weights (V-9), which must move NRMSE by more than one seed
sd. Mirror kill: `γ̂ → 1` sends `1/(1−γ̂)` to infinity and the certificate is vacuous — print
`1/(1−γ̂)` beside every `δ`, as `1/(1−â_max)` was printed and found undefined at all eight R1 seeds
(`V15_R1.md:56`).

**Sentences.** Holds: *"One learnable scalar interpolates next-step (softmax, bitwise at `γ = 0`)
and equilibrium (absorption) reads; on BED-S `γ̂ = [g]` (LR `Λ = [Λ]`, MOVED) reads `[k]` of
`t* = 8` hops in one solve."* Counter: *"The horizon dial is a definition; on BED-S training pins it
at `0` and the shape is softmax on that bed."*

**Mechanisms.** Ruling 10′; V-9; P-8 (`1/(1−γ̂)` printed); D-3 (`γ` seen to vary across
`t* ∈ {2, 8, 32}`).

---

## 3. The eleven NOT-FOUND components, one bet each

The first eight are C1's (e), (g), (h), (i), (j), (k), (l), (m); the last three are filed NOT FOUND
by the sweeps but not in C1 (`sweep_causality.md` §3.3 items 2, 3, 5; `sweep_safety.md` §2
"committor label class"; `sweep_expressivity.md` §3.6).

### 3.1 (e) `K ≥ 2` constraint sets plus a goal set as boundary rows inside the causal read
- **Prediction.** The identity rows change the *support* of the mixing matrix (mass that would
  have reached later positions is captured at the boundary) and that separates the shape from
  ChaCAL-with-sink by more than the MDE; the influence-Jacobian min entry reads exactly `0` (C3),
  predicted and printed.
- **Counter.** ChaCAL-with-sink matches within the TOST margin at `N = 70`; or the redirection is
  invisible at `t* = 8` because the boundary lies past the trained dial's horizon `γ̂^{t*}`.
- **Killer.** §2.2's second bet: `≈ 34 s` first, `≈ 228 s` to close; fires at TOST equivalence.
- **Sentences.** Holds: boundary rows are the mechanism (§2.2). Counter: *"boundary rows are a
  parameterisation of ChaCAL's sink; the shape is ChaCAL with a certificate."*
- **Mechanism.** V-24 (ChaCAL + same `γ` is the plant); V-12 (goal set); C3 (never "veto").

### 3.2 (g) the interventional re-solve `do(a)` with `Δz` as a trained, jointly scored vector
- **Prediction.** §2.1's; and the Sherman–Morrison pricing (`sherman-1950-inverse`,
  `hager-1989-updating` [V]; `sweep_safety.md` §1.A): one candidate costs one column of
  `(I − γP)^{-1}` plus an `s·d` inner product, so the `m = 8` sweep adds `≤ 2 × 1.041 ms` per
  step at `n = 2048`.
- **Counter.** The cosine does not separate (§2.1), or the measured increment exceeds
  `8 × 1.041 ms` — a re-solve per candidate is what ships.
- **Killer.** §2.1's `≈ 34 s`; the price check is a microbenchmark, seconds.
- **Sentences.** As §2.1; under the price counter: *"the consequence channel is priced as `m`
  solves, not one solve plus `m` updates."*
- **Mechanism.** M-8 / P-8 (increment measured at the shape's geometry, stated as a floor); V-24
  (`V = 1 ⇒ Δz ≡ 0`, RUN, rejection region `1.127`).

### 3.3 (h) per-constraint committor reads and the safest-move rule over in-context moves
- **Prediction.** §2.2's P1; and the Chebyshev and `argmax q^{(0)}` forms *disagree* on a non-zero
  fraction of draws (printed at construction), so both may be named.
- **Counter.** Disagreement fraction `0` (one form; `sec_refuted.md` C4) or argmin-unique fraction
  outside `(0.05, 0.95)`.
- **Killer.** The construction census, 0 GPU-s; fires at `sd(q^{(k)}) = 0` for any `k`,
  disagreement `0`, or the argmin-unique band violated.
- **Sentences.** Holds: two rules reported with their disagreement fraction. Counter: one rule
  named; the other column deleted.
- **Mechanism.** V-8 / V-12; D-3 (`t* ∈ {2, 8, 32}`, `K ∈ {2, 3, 4}`, `m ∈ {4, 8, 16}`; `m = 2`
  refused, V-10).

### 3.4 (i) a printed Neumann certificate with a planted negative (L-CERT)
- **Prediction.** On the *shipped mask* over 1,024 drawn cells, `‖O_full − O_mask‖_∞ ≤ δ·‖V‖_∞`
  with `δ = γ̂^{K+1}/(1−γ̂)` in vector units (`sec_cost.md` §4.x.3: RUN `6.13e-05` against bare
  `δ = 1.526e-05` and vector bound `≈ 7.6e-05`), and `δ < sd(label)`.
- **Counter.** Exceeded on any draw (a compliant kernel reads as a violation if `δ` is in the wrong
  units, V-17); `δ ≥ sd(label)`; or the planted non-stochastic `P` (rows `1.5`, err `119.37` vs bound
  `1.143`, RUN by the coordinator) is moot because the domain census finds `row-sum(P) ≠ 1` on drawn
  cells (V-25).
- **Killer.** 1,024 forward passes at `n = 2048`, `≈ 2.6 s`; fires on one exceedance.
- **Sentences.** Holds: *"every truncated read ships `δ·‖V‖_∞`; worst excess `[x]` below it on
  1,024 cells; the planted `P` exceeded its bound by `[y]×`."* Counter: the mask is refused
  (L-CERT); the exact solve, cheaper than one hop here (`2.514` vs `3.001 ms`, RUN), is the path.
- **Mechanism.** V-3 / V-10 (the bound is attained on this class, `sec_refuted.md` C6, so the pass
  is *declared definitional* and the bind is carried by the plant and the shipped-mask read);
  V-17; L-CERT.

### 3.5 (j) a learnable discount `γ` in a causal LM read, pinned by the LR test
- **Prediction.** On a language cell (enwik8 slice, pinned, `kaggle/README.md:31-43` — **not
  launchable by any agent; the author's yes is a node**), `γ̂` is MOVED on `≥ 6/8` seeds and the
  perplexity delta against ChaCAL at fixed `γ = 0.9` is inside the MDE.
- **Counter.** PINNED at `0` on `≥ 6/8` seeds, consistent with ChaCAL's own LM result being worse
  than its baseline (perplexity `21.46` vs `20.15`, `sweep_resolvent.md` §2.1; digits `[U]`).
- **Killer.** One LM pair; **price NOT MEASURED** (no LM cell of the shape has run). Until then the
  bet is placed on BED-S's `γ̂` (§2.3), `≈ 34 s`.
- **Sentences.** Holds: *"`γ` is a trained scalar; the LR test moves it off `0` on `[k]/8` LM
  seeds."* Counter: *"on the LM read `γ` pins at `0`; the shape's LM cell is softmax; `γ` is a
  bed-side dial."*
- **Mechanism.** Ruling 10′; V-9; `sec_refuted.md` C5; the Kaggle-yes memory rule.

### 3.6 (k) machine-checked containment (three corners + `γ = 0`) and the zero-gate iff
- **Prediction.** The four `[M]` targets of `sec_proved.md` §2.8 (`gamma_zero_is_softmax`,
  `resolvent_fromBlocks`, `lower_triangular_isUnit`, `segmentation_blockdiag`) build with zero
  `sorry` and axioms `[propext, Classical.choice, Quot.sound]` only; the four `[S]` targets
  (`pathprod_is_chain_resolvent`, `committor_is_resolvent_read` (b)(c), `neumann_truncation_bound`,
  `resolvent_is_triangular_solve`) close in one file each of the `V15Source.lean` scale.
- **Counter.** `committor_is_resolvent_read` (b) — invertibility of `1 − Q` from a Perron
  certificate at `ρ < 1` — does not close in one file because `‖Q‖_∞ = 1` off the absorber-adjacent
  rows (`sec_proved.md` §2.8 (3)); I3 is carried `[S]` with its numeric instance (`0.0` on BED-1's
  real sets) only.
- **Killer.** `lake build`, 0 GPU-s; fires on a `sorry` or `sorryAx`.
- **Sentences.** Holds: *"six identities are theorems; I3 is `[M]`."* Counter: *"five are theorems;
  I3 is a numeric identity at `0.0` with an `[S]` target."*
- **Mechanism.** P-11 (no `[M]` tag cited before it builds — the record's `#18/#19/#22 [M]` with no
  declaration, `sec_proved.md` §2.0); V-25 (`Nilpotent` transfers only to regime N,
  `one_not_nilpotent`).

### 3.7 (l) the influence-Jacobian `β₀` barcode with a directed-stability `δ`
- **Prediction.** On BED-S the `β₀` barcode of `|∂z_i/∂x_j| > ε` has its `ε = 0` endpoint equal to
  the exact F0 segmentation count (`segmentation_blockdiag`), and a **null barcode** from
  row-permuted influence at the same logit scale (M-15) differs from the measured one by more than
  the printed `δ` from `turner-2019-quasimetric-rips` [V] at every `ε` in the frozen grid.
- **Counter.** Measured and null agree within `δ` at every `ε` — the barcode is a statistic of the
  logit scale, as the `kushnareva-2021-tda-attention` [V] lineage reads it, with no causal content.
- **Killer.** One forward pass and one Jacobian per seed, eight seeds; seconds. Fires at
  `max_ε |β₀ − β₀_null| ≤ δ`.
- **Sentences.** Holds: *"the influence barcode is separated from its permutation null by `[x]`
  at `[k]` of `[K]` thresholds, `δ = [d]` from directed stability."* Counter: *"the barcode is
  decoration; the F0 endpoint is the only certified topological statement."*
- **Mechanism.** M-15; M-2 (grid and `δ` fixed before the read, as `ceq/certs/topological.py:47-56`
  does); V-16 (non-integer `β₀` is a refusal); P-5 (the M10 ids corrected to
  `hofer-2019-connectivity-optimized`, `moor-2020-topological-autoencoders`,
  `carriere-2021-optimizing-ph`, `sweep_topology.md` §4.1).

### 3.8 (m) Mapper cover → causal CSR schedule → certified resolvent
- **Prediction.** At `s = 4096` a Mapper cover with parameters fixed by the Reeb-estimator rule
  (`carriere-2018-mapper-statistics` [V]) on a held-out relation, quantised to `[B, B]` tiles,
  yields a schedule whose Neumann read (`sec_cost.md` §4.x.5, union certificate) matches the dense
  solve within `δ_F1 + γ^{K+1}/(1−γ)` in vector units at a dividend `D = s²/Σ L_m² ≥ 4×` in visited
  tiles over the do-nothing 0D-salience schedule (`sharma-2026-kernels-22` [V], the standing control).
- **Counter.** The do-nothing schedule matches within the MDE at equal visited tiles (V-9), or the
  exact solve on the F1 mask fills in and the dividend is `< 2×`.
- **Killer.** **NOT MEASURED — needs a chunked kernel and the resolvent stage in `kernels#22`**
  (forward-only, `THEORY.md:223-224`). Until then the bet is on the *segmentation* dividend, exact
  and free: BED-M's `31.06×` (DERIVED from `96.78 %` zero-hop, `V16_ARM_SMPRIME.md:518-522`), a
  corpus property quoted with the corpus named (V-22).
- **Sentences.** Holds: *"a Mapper cover buys `[D]×` over the do-nothing schedule at `s = 4096` with
  a printed union certificate."* Counter: *"the candidate builder is the merged 0D-salience
  schedule; Mapper adds nothing measurable; the exact dividend is segmentation's."*
- **Mechanism.** V-9; V-22; L-CERT (exact solve on an F1 mask is refused); P-4 (no scaffolding
  claimed).

### 3.9 (n) the vector label `z*` / `Δz` scored jointly against a matched-depth per-row control on byte-identical draws
- **Prediction.** The harmonic residual `r(ẑ) = ‖(I − γP_env)ẑ − V‖_∞/‖V‖_∞` separates the arms by
  `≥ 10×` while per-coordinate mean NRMSE differs by less than the MDE (`sec_beds.md` §6.E P3).
- **Counter.** `r(q̂_softmax)/r(q̂_shape) ≤ 2` — joint consistency is learnable by a per-row mixture;
  `MATHEMATICS.md:106-127` is withdrawn. `(2, 10)` is SPLIT and reported as such.
- **Killer.** The same `≈ 34 s` pair; the residual uses the bed's operator as a *score*, never a
  loss (D-2). Fires at ratio `≤ 2`.
- **Sentences.** Holds: *"the shape's read is jointly consistent (`r = [x]`) where depth-1 softmax
  is not (`r = [y]`), at equal marginal error."* Counter: *"joint consistency is learnable by a
  per-row mixture; the one-read property is a cost statement, not a capability."*
- **Mechanism.** V-26 (marginal `W1` refuted: permutation reads `0.0` at NRMSE `1.421901`,
  `V20_R15_THEORY_TABLE.md:221`); D-2; M-2 (three-way branch fixed now).

### 3.10 (o) the EMC bind — order of intervention and equilibration
- **Prediction.** For the causal class "intervene then solve" and "solve then re-solve the suffix"
  coincide (**RUN this session**: `1.33e-15` at `s = 64, γ = 0.5`); a planted cyclic `P` through
  the intervened row reads a difference `> 0.1` in `‖·‖_∞` (`dash-2005-emc` [V] Thm 1).
- **Counter.** The planted feedback instance reads `< 1e-6` — no rejection region on the class the
  shape uses; EMC is filed as a definition, not a proposition (V-24).
- **Killer.** One float64 one-liner on a planted cyclic `P`; seconds; the cheapest bet here.
- **Sentences.** Holds: *"EMC holds for the causal class by nilpotency and fails on a planted
  feedback instance by `[x]`."* Counter: a remark with the Dash citation, no proposition number.
- **Mechanism.** V-24; D-7 (both halves filed); `momennejad-2017-sr` / `russek-2017-predictive`
  [V] as the external argument that a cached resolvent needs a re-solve under a transition change.

### 3.11 (p) the committor / reach-avoid *label class* for an attention bed, with floor and pricing
- **Prediction.** BED-S admits: all four guards of `sec_beds.md` §6.C.4 fire both ways (leak
  detector below `0.5` with `Q_a` rows planted, above `0.9` on local features, as E4′'s fixed point
  at `0.973819`); the realised paired sd of the first eight seeds is `≤ 0.034451`, so
  MDE `≤ 0.039827`; `t* = 8` lands on a bottleneck of `c` edges with `vol(S)/(2c)` of that order
  (`e4_harmonic.cheeger_t_rel_floor`, `:247-263`; the one-bridge case read `1372.50`).
- **Counter.** No admission: the realised sd is `2.18×` the pilot (M-3's precedent,
  `0.050146 → 0.109199`, MDE `0.126238`) and nothing in §2 is falsifiable at `N = 8`; or `t*` cannot
  be placed in `{2, 8, 32}` because the cut collapses the label (`sd 0.499989 → 0.038445`,
  `scale/e4_harmonic.py:425-434`).
- **Killer.** Construction plus eight seeds of one arm, `≈ 17.4 s`; fires at the sd read.
- **Sentences.** Holds: *"BED-S is a registered bed with exact oracle at `0.0`, Fano floors
  `[f_0, f_1]`, paired sd `[sd]`, MDE `[mde]`."* Counter: *"BED-S needs `N = [n]` for the `t* = 8`
  cell to be falsifiable; the paper files the bed and no capability number."*
- **Mechanism.** D-4; M-3; M-9 / V-5 (`REQUIRED_SEEDS = 8`, deduplicated); D-3; L-FLOOR (never
  `floor₁`).

---

## 4. Identities RUN this session (the free binds, 0 GPU-s)

| id | statement | number | rejection region |
|---|---|---|---|
| C6 | `Δz = (I − γP')^{-1}(ΔV + γΔP z)`, causal softmax `P`, `s = 64, d = 16, γ = 0.5`, row 20 → `e_5` | `1.36e-15` | carried by the plants below |
| C6-plant | `V = 1 ⇒ Δz ≡ 0` for a row-stochastic row change | `0.0` exactly | random `V`: `1.127` |
| C6-causal | `Δz` zero on every row before the intervened position | `0.0` on rows `< 20`; `1.127` on rows `≥ 20` | — |
| EMC | intervene-then-solve vs solve-then-re-solve-suffix, triangular `P` | `1.33e-15` | the planted cyclic `P` of §3.10 (NOT RUN) |
| C5 | `Σ_k q^{(k)} = 1` on `T` with `K = 2` sets only | row sums `[0.99999999999999978, 1.0]`; `min_i max_k q_k = 0.612 ≥ 0.5` | a goal set makes `Σ_{k≥1} < 1` |
| Fano | zero-hop floor `1 − ln 2/ln m` | `0.5 / 0.6667 / 0.75` at `m = 4/8/16` | `m = 2` gives `0.0` (refused) |

I1–I5 (`BRIEF.md` §1) and JUPITER's corrected I3 (`0.0` on BED-1's real sets, `sec_proved.md`
§2.8 (3)) are inherited, not re-run.

---

## 5. Ranking by probability of dying, and the death order

Ordinal (D-CALIB-3). The evidence column is what the record already holds *against* the
component; the order is the order the kills run (D-CALIB-4).

| rank | component | evidence the record already holds against it | killer | survives its death |
|---|---|---|---|---|
| 1 | **(p) BED-S admission** (§3.11) | every bed died or stalled at construction: BED-M's first builder leaked `1/(t*+1)` at zero hops, three of five rungs aborted (`scale/negation_scope.py:399-415`); BED-K's shape never run (`V20_R15_LEAP_LEDGER.md:323`); BED-1 cells `0`; E4′'s label collapses under a kill rate; BED-M's realised sd was `2.18×` the pilot (M-3) | `≈ 17.4 s` | **reroute** to `bed_1.build(jitter)` with `K = 2` and `B` as goal — oracle, guards, Morse census and CK test exist and are tested (`sec_measured.md` M.7.2); **reprice** `N` from the realised sd |
| 2 | **(j) `γ` pinned at 0** (§2.3, §3.5) | hop-2 sweep: no `γ` beat `0`, 4/4 blind predictions held (`V13_PREDICTION_HOP2.md:35-46`); `pivot_signed` was `pivot_unsigned` wearing a name (`workdonenew.md:379`); ChaCAL's LM result is worse than its baseline | `≈ 34 s`; LM NOT MEASURED | **retire** the LM `γ` claim; **keep** `γ` as a bed dial and regime N (`γ = 1`, nilpotent, corner 3), exact without a dial |
| 3 | **(e)/(h) ChaCAL-with-sink matches the committor read** (§2.2, §3.1) | the two-branch escape passed parity bitwise for the label itself (V-24, `workdonenewseal.md:124-126`); sinks already act as key biases storing non-informative mass (`gu-2024-sinkemerges` [V]); Zhu 2003's clamped nodes are the same rows, 23 years old | `≈ 34 s`, then `≈ 228 s` | **retire** boundary rows as a capability; the shape is *ChaCAL + certificate + Lean containment* — a property paper about a published operator |
| 4 | **(n) residual does not separate** (§3.9) | the claim is recorded UNTESTED (`MATHEMATICS.md:106-127`); `duranthon-2026-softmaxadvantage` [V] Prop 4.2 says the marginal will not separate and nothing says the residual will | `≈ 34 s` | **reprice**: one-read is a *cost* statement (one solve, depth `s`); the residual stays as a diagnostic |
| 5 | **(a) consequence field** (§2.1, §3.2) | softmax reads sign fidelity `0.807843`; `zhang-2023-cina` [V] | `≈ 34 s` | **retire** the sentence; **keep** C6 and the `V = 1` plant |
| 6 | **(l) barcode is decoration** (§3.7) | half the Euler–Poincaré check cannot fail (`workdonenewseal.md:265`); `0.7549` had `z = −0.30` against its null (M-15) | seconds | **retire** to the F0 endpoint, a theorem |
| 7 | **(m) Mapper adds nothing over do-nothing** (§3.8) | the do-nothing schedule is the standing control by design (`THEORY.md:162-165`); `index_select` `2.58 ms` vs `0.82 ms` attention (`ceq/multizoom.py:12-16`) | NOT MEASURED | **reroute** to segmentation's exact dividend |
| 8 | **(i) certificate exceeded on the shipped mask** (§3.4) | the bound is attained on this class (`sec_refuted.md` C6); units were V-17 once | `≈ 2.6 s` | **retire** the mask; the exact solve is cheaper than one hop |
| 9 | **(o) EMC plant has no rejection region** (§3.10) | none; the triangular half is RUN at `1.3e-15` | seconds | **retire** to a remark |
| 10 | **(g) priced as `m` solves** (§3.2) | the FLOP model understated a bill by `2×–3×` (`MISTAKES.md:469-476`) | seconds | **reprice** |
| 11 | **(k) I3 (b) does not close in one file** (§3.6) | `‖Q‖_∞ = 1` off absorber-adjacent rows; `#14 committor` moved `[S] → [D]` across two contracts | 0 GPU-s | **keep** the `0.0` identity, carry `[S]` |

**Where the shape most likely dies first.** At rank 1, **before a GPU-second is spent**: BED-S
does not admit at `t* = 8` because the realised paired sd is too large for `N = 8`, or because a
bottleneck that places `t*` at 8 collapses the label. That is the record's dominant failure shape
(D-4, M-3, V-8) and it orders the killers. The death is cheap (`≈ 17.4 s`) and survivable: the
replacement route is the jittered `bed_1` landscape with `K = 2` and goal `B`, whose oracle,
residual (`4.16e-17`), CK test, Pesin deficit and Morse census are in the tree and tested (13
tests, 5 mutants killed, `sec_measured.md` M.7.2), repriced to the `N` its realised sd demands.

**The second death is the expensive one.** If BED-S admits, rank 3 — ChaCAL-with-sink matching
the committor read at `N = 70`, `≈ 228 s` — removes the *only* NOT-FOUND component that is a
mechanism rather than a discipline. What survives is a paper whose contributions are five
machine-checked identities on ChaCAL's operator, a certificate discipline with a planted
negative, a bed with printed floors, and the negatives: a *property paper about someone else's
operator*. The paper must be written so that it is still honest in that state.

**Dependencies.** Ranks 1–3 are not independent: rank 2 (γ pinned) makes rank 3 unrunnable
(ChaCAL at `γ̂ = 0` *is* softmax and the committor read is `PV`). The critical path: construction
census (rank 1, free) → one `≈ 34 s` pair reading `γ̂`, the argmin CP, the residual ratio and the
field cosine at once (ranks 2, 4, 5, and 3 at `N = 8`) → the `≈ 228 s` TOST only if `γ̂` is MOVED
and the `N = 8` reading did not separate. **The whole falsification programme on BED-S at the
design point is under five GPU-minutes** (`17 + 34 + 228 + 2.6 s ≈ 4.7 min`) plus free CPU
identities and Lean files; the `≈ 36 min` ladder is spent only on a survivor.

---

## 6. The two honest sentences

**6.1 If every prediction holds** (the optimistic half — the half the calibration record says is
the less likely one):

> *Consequence–Equilibrium Attention is ChaCAL's causal resolvent read (`fagnou-2024-chacal`) on
> the record's Lean-checked three-corner base, with `K + 1` absorbing sets as boundary rows. On
> BED-S (`t* = 8, m = 8, K = 2` + goal, `N = 8`, 4,769 parameters) it reads the reach-avoid vector
> with argmin error `[e]` (CP `[l, u]`) below the one-hop Fano floor `[f]`, harmonic residual `[r]`
> against depth-1 softmax's `[r']` (ratio `≥ 10×`) at equal marginal NRMSE, and a displacement
> field with cosine `[c]` against `[c']` (McNemar `p = [p]`); ChaCAL with the same `γ̂ = [g]` (LR
> `Λ = [Λ]`, MOVED) and a sink token reads `[e'] > f`; the depth-5 softmax skyline reaches the same
> argmin accuracy within the MDE, so the shape's separate advantages are exactness (printed
> `δ·‖V‖_∞`, worst excess `[x]` on 1,024 cells), one-read joint consistency, and the boundary-row
> mechanism — at `+50 %` causal MACs and depth `s` over the softmax head it contains bitwise at
> `γ = 0`.*

Every bracket is a number BED-S has not produced; the sentence is a template with its floors,
controls and prices in place, and it is the *most* the tables could ever permit.

**6.2 If the median outcome holds** (every counter is the point estimate, D-CALIB-1):

> *Consequence–Equilibrium Attention is a re-parameterisation of ChaCAL's causal resolvent read
> with absorbing boundary rows on the record's three-corner base. Six of its identities are
> machine-checked and the seventh (the committor as a resolvent read) holds numerically at `0.0`;
> its Neumann certificate ships with a planted negative that fires (`119.37` vs `1.143`); BED-S is
> registered with an exact oracle at `0.0`, restricted-view Fano floors and a printed paired sd.
> On BED-S at `N = 8` its trained `γ̂` is not distinguishable from `0` by the LR test, ChaCAL with a
> sink token is not distinguishable from the boundary-row read, and depth-1 softmax matches its
> marginal NRMSE, residual and displacement cosine within the MDE. No capability sentence is
> licensed; the contributions are the identities, the certificate discipline, the bed, and the
> negatives.*

That is the sentence the author should read first. It is what the record's own calibration says
the tables will most likely permit, and it is still a paper: fifteen rounds shipped no attention
claim and 169 theorems; 6.2 ships seven identities, a bed and a control discipline on an operator
the field has now published twice without either.

---

## 7. What is not bet on

No bet on the resolvent read, the Neumann series, the absorbing-chain reading, the committor
identity, the successor representation, the rank-one re-solve or the reach-avoid rule — each is
OCCUPIED with its owner in §0.3. No bet against the deeper / wider / CoT skyline
(`sanford-2024-logdepth` Thm 4.2, `yehudai-2025-depthwidth`, `merrill-2024-cot`, all [V]); it is
the honest control and `wang-2024-incontext-td` / `xie-2026-softmax-rl` [V] show it computes the
same resolvent by iteration; "beats softmax" and "beats native" are banned (R-SKY;
`sec_refuted.md` Part D). No bet on any BED-M cell: BED-M is *contained* (I2, `0.0` entrywise) and
a shape reading on it is a reproduction check (D-2), kept only as the parity bind and the
truncation-law must-fire. No bet whose killer needs code that does not exist without saying so
(§3.5, §3.8). No bet on Kaggle: launch requires the author's explicit yes (`kaggle/README.md:9-11`).

---

## 8. Limits (collected once)

All RUN numbers in §4 are one seed, one CPU box, float64, identity checks without intervals. Cell
prices are the `sec_cost.md` §4.x.8 assembly: fitted laws at `s = 64` only (D-3: the exponent in
`s` is unidentified on `40/40` banked cells), a per-op solve increment that is a floor, a laptop
clock that is not stationary (±12 %), and no end-to-end shape cell ever timed. The depth-5 skyline
price is ASSUMED linear in depth; the InfSA-style control price is a floor from the `K = 16`
microbenchmark. BED-S has no cell, no realised sd, no measured `t*`, no `I(X_{≤k}; a*)`; every
BED-S number here is a floor formula or a design constant. The LM `γ` bet and the Mapper bet have
no runnable killer in the tree. The ranking in §5 is ordinal; no shrink factor is authorised. The
`[V]` marks are the sweeps' (abs page or registry record, title matched); ChaCAL's perplexity
digits and its "geometric, not nilpotent" remark are `[U]` summariser readings. The Misra 2023
multichain warning is carried to the paper's Limits, not resolved here. No code file was written;
no git write was made; no hardware run beyond the six one-liner identities.
