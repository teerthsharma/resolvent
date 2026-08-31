# CEQ v15 — THE COMPOSITION ROUND

Contract of record for R11. Absorbs v14–v14.3 and X32. Supersedes everything
after `attic/workdonenew.pre-v13.md`. Filed verbatim from the author's message
of 2026-08-31 so that every clause below is citable by file and line rather
than by memory of a prompt.

---

## NORTH STAR (immutable)

Attention that is EQUAL to self-attention on its own ground, built FROM softmax
and AdamW, and capable on ground they cannot occupy — it predicts the NEXT
STATE toward equilibrium (which basin, whether at a decision point, which
transition), not the next token.

**Distance to this sentence is the last line of every iteration report.**

---

## WHERE WE ARE (binding, from the pre-v13 census)

- softmax and `pivot_unsigned` share a BIT-IDENTICAL operator; the entire
  architectural difference was the hop-2 term.
- Four constructions of that term (routed, scaled, Wiener-deflated, gated) all
  fail to reach the 1-hop baseline.
- The Wiener analysis shows the term carries no usable signal at any mode,
  trained or untrained.
- No arm at any cell has crossed floor-1 (h-hat <= 0.389, full census).
- C-PAR TOST at N=8 is unreachable by arithmetic: the CI first fits at N=23,
  power 0.8 at N=70.
- The sizing line was 6.63x low.
- Every corpus to date is Markov-generated — no label has ever contained a
  delayed cause.

**THE ONE SURVIVING ALGEBRAIC FACT.** The label's t-hop term is a PATH PRODUCT
`a_{s-1} ... a_{s-t} . b`. Additive attention hops compose weights across
positions; the label composes values along paths. Same word "hop", different
algebra.

---

## STANDING LAWS

All paid for: L-VALUE, L-SCOPE, L-BIND, L-PRIOR (fetched EQUATIONS, never
keywords), L-G2, L-FIRST, L-TOL, L-TIME, D-1 (work is a DAG; parallel dispatch
only on no-shared-state nodes), D-2 (skills are modes), D-3 (no loop mounts
before `ecbedf7` + `MISTAKES.md` + the 15 loop failures are read and the old
loop's cause of death is stated in one sentence).

**NEW — L-EQ.** The tag `[V]` (page exists, intro matches) is INADMISSIBLE for
any load-bearing statement. `[V-eq]` requires the theorem's statement WITH
HYPOTHESES and one numeric instance run before the statement enters a contract.
Author's own defect class; two-thirds of the pre-v13 section-5 strikes were
`[V]`-as-theorem. Filed under the author's name.

**NEW — L-LEAN.** The arm may not be TRAINED before its identity theorems are
green. Binds become theorems; tests confirm theorems.

---

## THE ROOM

| Planet | Owns |
|---|---|
| JUPITER | maths, Lean, fetches |
| SATURN | instruments, controls, manifests, `MISTAKES.md` |
| MERCURY | pricing (RULE 4 same-message critical path), scheduling, execution, tables |
| MARS | standing adversary, at most 1 value-testable attack per iteration |
| VENUS | one competing numeric prediction per deciding measurement, committed before data |
| NEPTUNE | systems gate — dispatch, memory by the CALIBRATED sizing model `C_OPERATOR = 3.9 x 3.4 B/elt`, wall-clock vs FLOP |

Planet names label responsibilities, not processes (D-2).

---

## PART I — THE ARCHITECTURE, AS THE ALGEBRA NOW FORCES IT

**THEORY OF THE COMPOSITION** (Mori-Zwanzig `[V-field]`; projection of history
onto a low-dimensional state yields Markov term + MEMORY KERNEL + noise):
neither a recurrence nor attention is the architecture; the composition is
forced by the projection theorem.

### S-M — MARKOV CARRIER (the prefix-logit form)

Learn `g(x_i) = log a_i` with `g = -softplus(W x)` (`log a <= 0` implies
stationarity by theorem #6); `C = scan(g)` (prefix sum, `O(s)`, Lean #7
licenses the parallel route); ONE unnormalized causal hop
`W_ij = exp(C_i - C_j)` on values `V(x)`.

Signed variant: parity mask `(-1)^{P_i XOR P_j}`, `P` = prefix-XOR of sign bits
(Lean #3).

`[RUN]` reproduces the chain label to `4.0e-15`; signed to `3.1e-15`;
`g == 0` gives the causal all-ones mask = standard attention.

General DAG: the PROVED resolvent `(I - A)^-1` fed EDGE GATES, not similarities
`[RUN: matches brute-force path sums to 1.1e-16]` — and this IS the successor
representation with a learned world model (F3-F5), no longer a slogan.

**Alternative carrier S3-R** (replicator-mutator, X32, CORRECTED): injection
`mu . softplus(W_b x)` (Poincare-Hopf needs `mu > 0`; a learned linear injection
can point outward); `rho_P` gate at **2** (antisymmetric `J` gives 2 EXACTLY;
`sqrt(2)` was the random-matrix value); `dP/dt = Var(f)`, `f-bar = 2P`; the
mechanism is normalization x injection, not normalization alone.

### S-K — MEMORY KERNEL

Attention, kept exactly as it is, plus the fractional-order head (X34):
Grunwald-Letnikov weights `w_k = (-1)^k C(-alpha, k)` parametrize a power-law
kernel by ONE scalar `alpha` (Riemann-Liouville `[V-field]`); binds
`alpha -> 0` identity, `alpha -> 1` cumulative sum, bitwise on fp64 probes.

### S-G — GUARDS

The itinerary of isocommittor crossings (`q = 1/2`, TST's dividing surface, the
proved committor object) IS the action vocabulary; next best action = next
symbol. Pesin's identity as admissibility (X33): `DEFICIT = lambda-hat - h_sym`,
approximately 0 iff the guards are a generating partition
`[RUN: 0.0003 at the right guard, 0.156 at a wrong one]`. On token streams the
deficit is the number behind "Markov hinders inside a chat".

### S-C — CONSERVATION CENSUS

Carriers conserve mass to `1e-12` (softmax rows 1.000; replicator by Lean #8);
readouts print their dissipation budget (GELU passed 0.519 — and that
dissipation is where the sign came from; unaccounted dissipation is the defect,
not dissipation).

### CLAIM SHAPE, fixed

Composition only. Components are OCCUPIED and cited before they are named:
SSD/Mamba-2, GLA, RetNet (the decay mask); negative-eigenvalue SSMs (signed);
MWU/Hedge (replicator); Mori-Zwanzig-in-ML (learned kernels); ARFIMA/fractional
nets; symbolic-dynamics / generating-partition estimators; Hamiltonian /
flow-conserving nets. JUPITER fetches all at equation level in the round's first
node.

**PARITY WITH SELF-ATTENTION is by IDENTITY BIND, not TOST.** `g == 0` gives
bitwise standard attention (Lean #5). TOST retires to `N >= 23` runs; below
that, resolution statements only: "excludes a difference beyond
`Delta = t_{.975, N-1} . sd / sqrt(N)` and nothing smaller".

---

## PART II — THE LEAN INVENTORY (prove before train)

| # | statement | status |
|---|---|---|
| 1 | `chain_path_product` | [M] |
| 2 | `prefix_logit_mask` | [M] |
| 3 | `parity_sign` | [M] |
| 4 | `resolvent_eq_path_sum` | [S] |
| 5 | `gate_zero_is_attention` | [M] |
| 6 | `bounded_gates_stable` | [M] |
| 7 | `scan_assoc` | [M] |
| 8 | `replicator_eq_cumsoftmax` | [M] |
| 9 | `walsh_two_point` + `multiplicative_epistasis_nonzero` | [M] |
| 10 | `hop_floor` | [D] |
| 11 | `lyapunov_gate_stationarity` | [D] |
| 12 | `first_order_cannot_delay` | [S] |
| 13 | `GL_weights_powerlaw` | [S] |
| 14 | `committor_eq_harmonic` (from v-main.7) | [S] |

**Train-gate:** #1, #2, #5, #6, #7 green before ARM PL trains; #8 before S3-R;
#12 before BED-K's scan verdict is called a proof. Eight `[M]` items,
approximately one Jupiter week.

---

## PART III — THE BEDS

Two-sided native test; battery C-A..C-F + CK test + Morse census + conservation
census on all.

- **BED-M** (Markov, exists) — the chain corpus, scan-native. Skyline = the
  gated scan at 4,769 matched (`m = 93`). Floors `sqrt((t* - h) / t*)`.
- **BED-K** (memory kernels, new) — `z_i = sum_j K(i, j; x) b_j` with (a) pure
  delay `d`, (b) power-law `K` (fBm-type, `H > 0.5`) — attention-native,
  scan-blind by Lean #12 `[RUN: best first-order recurrence 0.990]`. The Hurst
  covariate ships ONLY with its white-noise must-fire (the author's R/S read
  0.75 on AR(0.5): biased; Jupiter picks a calibrated estimator).
- **BED-1** (multi-basin, v-main.7) — labels by SPLITTING PROBABILITY
  (committor), never by barrier height. The pre-v13 strike: parallel saddles
  make the higher barrier the faster channel above `T* = DeltaDeltaE-dagger /
  ln m` `[reproduced to 6.4e-16]`. Guards = `q = 1/2` surfaces; itinerary =
  actions; Pesin deficit printed.
- **INTERVENTIONAL CHANNEL** (all beds) — the `do()`-bit extended to
  per-position bumps with oracle responses; gates are JACOBIANS
  `[RUN: recovered to 1e-9 from one bump]` — `L = L_obs + zeta . L_jac`.
  Genomics' lesson (Perturb-seq / GRN `[V]`): interventions identify `a`
  directly; observations confound `a` with `b`.
- **EPISTASIS positive control** (X8 reinstated) — multiplicative landscapes
  have nonzero Walsh degree-2 by Lean #9 `[RUN: 0.0375 vs 0.0000 additive]`;
  the probe reads the oracle's coefficients or it is broken.

---

## PART IV — THE DECIDING MEASUREMENTS

L-FIRST; Venus files before each; `N = 8` seeds; resolution statements; every
cell with manifest, floor columns, distance-to-native-skyline, conservation
drift, Lyapunov `lambda-hat`, unit-root flag.

- **R1** BED-M, `t* = 2`, `n = 2048`: ARM PL vs softmax vs scan-skyline.
  PREDICTION: `PL < floor_1 = 0.7071` with CI — the first floor crossing in the
  campaign. Bind first: oracle gates in implies label to `<= 1e-6`. Kill: bind
  passes, floor not crossed implies `g` unlearnable at budget — diagnose by
  linear probe on `log a` (should be near-exact), never by a new construction.
- **R2** BED-M, `t* = 8`, `n = 32768`: `PL < floor_1 = 0.9354`, `h-hat > 1` for
  the first time.
- **R3** BED-K delay bed: scan-only `>= 0.95` (theorem-backed), attention /
  fractional head near 0, COMPOSED arm matches the native primitive. BED-K
  power-law bed: `alpha-hat` recovered, `H-hat = alpha-hat + 1/2` within CI of
  the calibrated estimator.
- **R4** THE TWO-SIDED PARITY TABLE: the composed arm within `Delta_res` of the
  native skyline on BOTH BED-M and BED-K — the honest architecture sentence; no
  single-bed skyline can fake it.
- **R5** BED-1: S5' exit accuracy with committor labels, CK test
  `T-hat(n.tau) ~= T-hat(tau)^n`, Morse census, Pesin deficit on the `q = 1/2`
  guards, conservation census; the `(K-hat, Pesin-deficit)` policy: point-predict
  where generating and dominant, distribution-predict otherwise, refuse where
  both instruments expire.
- **R6** Interventional ablation: `zeta > 0` vs `zeta = 0` at fixed `n` —
  prediction: at least 2x fewer examples to R1's crossing with `L_jac`.

**MARS's standing attacks** (filed at it.0, bound by Saturn): floor crossing by
feature leak (Neumann-term census on the reading tensors); parity by
underpowering (achieved-power column); the skyline leaks the oracle (gate values
never in `x`); guards chosen after seeing itineraries (journal timestamp
ordering).

---

## PART V — THE DAG (D-1) AND THE SCRIPT

**PARALLEL-SAFE AT it.0** (no shared repo state):
- `n1` all component prior-art fetches at equation level (Jupiter)
- `n2` the D-3 read (Saturn)
- `n3` Lean #1, #2, #3, #5, #6, #7 (Jupiter, separate branch)
- `n4` BED-K + interventional-channel generators (Cameron/Saturn, fresh files)

**THE CHAIN (strict):** Lean train-gate green -> ARM PL built + binds
(oracle-gates, `g == 0`, DAG resolvent) -> R1 -> R2 -> skyline + composed arm ->
R3 -> R4 -> BED-1 heads + guards -> R5 -> R6 -> tables -> package -> prognosis.
Critical path is approximately 24 nodes; a loop mounted post-D-3 gets its length
from this count.

| iterations | work |
|---|---|
| it.1-4 | `n1`-`n4` in parallel; `MISTAKES.md` gains L-EQ with the section-5 census as evidence; the sizing model replaced by the calibrated one everywhere it was cited |
| it.5 | Lean train-gate verdict (#1, #2, #5, #6, #7 green, or the failing statement named — a failing `[M]` item means the arm is WRONG, and that is the point) |
| it.6-7 | ARM PL + three binds; identity manifests; Neptune's memory line from the calibrated model |
| it.8 | Venus files R1/R2; Mars files attacks; Mercury prices |
| it.9-10 | R1 then R2 (the deciding cells); verdicts by floor with CI; resolution statements |
| it.11 | Branch: crossing implies every prior hop arm retired in one commit with the correlation table (0.10 vs 1.00) as epitaph; no crossing implies the linear-probe diagnosis, then stop and report |
| it.12-14 | scan skyline (`m = 93`, identity bind `<= 1e-6`) + composed arm; BED-K registered past its battery (Hurst must-fire seen) |
| it.15-17 | R3, R4 — the two-sided table; conservation census on every arm; Lyapunov / unit-root columns |
| it.18-21 | BED-1 committor labels, guards, S5' head, X33 deficit, CK + Morse; R5 |
| it.22-23 | interventional channel live; R6 |
| it.24-25 | Lean #4, #8, #12, #13 landed or priced; #10 / #11 / #14 status stated |
| it.26-27 | tables cut (softmax first, skylines second, everything with its columns); HF package on the author's say-so |
| it.28 | Mars's fifteenth-class hunt on the trained artifacts; Venus's forecasting record |
| it.29 | `MISTAKES.md` round entry: the wrong-hop mechanism sentence ("the label composes values along paths; the arms composed weights across positions"), L-EQ, the Markov-corpus blindness, the six author-owned strikes |
| it.30 | Prognosis: scoreboard, the one claim sentence the tables permit, next-X list. STOP |

---

## SCOREBOARD

Moves only at it.10, it.17, it.21, it.23, it.30.

| event | points |
|---|---|
| Lean train-gate green | +2 |
| R1 floor crossing (first ever) | +12 |
| R2 `h-hat > 1` | +4 |
| BED-K registered | +2 |
| R4 two-sided parity | +8 |
| R5 state model passes CK with committor labels | +4 |
| R6 intervention budget halved | +2 |
| conservation + Lyapunov columns on every table | +1 |
| package | +3 |

Ceiling approximately 38 from the current 22 percent.

**Floor:** a diagnosed non-crossing with theorems attached, and D1's strongest
chapter — "additive attention cannot form path products; here is the census,
here is the construction that can, here is where it too is blind."
