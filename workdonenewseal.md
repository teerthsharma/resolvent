# workdonenewseal — status report

**Every number in this document carries its source. Where a number has no live
producer it is marked NOT MEASURED rather than estimated.**

State as of `91b862d`, 2026-08-31. **45 commits** this session across two rounds
(R11 "the composition round", v15/v15.1/v15.2; R12 "the banking round", v16, in
progress at it.6 of 40).

---

## 0. THE ONE-LINE ANSWER

**A composition claim that is earned, a capability claim that is not, and a
failure taxonomy that is now the repository's largest verified asset.**

An attention operator exists that provably contains softmax, linear attention and
the exact path product as three corners of one family; its identity binds hold to
`5.9e-16` on the corpus's real support; and on the deciding measurement **five of
eight seeds crossed a floor no arm in this campaign had ever crossed, three
diverged, and the confidence interval straddles.**

---

## 1. THE CLAIM LADDER

The shipped sentence requires `C-PAR ∧ (C-CAP ∨ C-TS)`.

| bar | demands | status | deciding measurement |
|---|---|---|---|
| **C-PAR** | parity with softmax at matched params | **INSTRUMENT REPAIRED** | TOST closed (`0.8807σ` half-width at N=8 vs a `0.5σ` margin; power `0.80` first at N=70; `0.0669` at N=23). Identity bind refuted, then repaired: bitwise softmax at the `β=1, g≡0, QK-on` corner, row sums `1.000000` |
| **C-CAP** | seed CI below `floor₁ = √((t*−1)/t*)` | **NOT EARNED** | R1: 5/8 crossed, CI `[0.617075, 1.041227]` straddles `0.7071067812` |
| **C-TS** | transition-state / exit accuracy | **BED BUILT, NOT RUN** | BED-1 exists, 28 tests; R5 not run |

**Scoreboard: R11 closed at 4 of 38. R12 open at 0 of 41.**

---

## 2. MACHINE-CHECKED MATHEMATICS

**12 Lean files · `lake build` exit 0 at `[1530/1531]` · 0 `sorry` · 0 `sorryAx`.**

Exit 0 is not treated as sufficient — a `sorry` inside a macro still builds — so
**every theorem is run through `#print axioms`** and depends only on
`[propext, Classical.choice, Quot.sound]`.

| file | round | carries |
|---|---|---|
| `Contraction`, `Occupancy`, `Nilpotent`, `OrbitBound`, `Refcount`, `OracleSeparation` | pre-R11 | the resolvent identity `M = (I − γP)⁻¹ = Σ γᵗPᵗ`, exact in `n` terms |
| `V15.lean` | R11 | train-gate #1,#2,#3,#5,#6,#7 — **and the refutation** |
| `V15Fork.lean` | R11 | the repair; `no_row_stochastic_with_drive_values` |
| `V15Kernel.lean` | R11 | #12, #13; two naive readings proved FALSE |
| `V15Source.lean` | R11 | #15; `inverse_identity_is_vacuous` (no axioms, used by nothing) |
| `V15Phase.lean` | R11 | #16; `six_never_reaches_the_band`; `lru_modulus_lt_one` |
| `V16Domain.lean` | R12 | #2 re-stated on real support; #5a, #5b; domain censuses |

### 2.1 The load-bearing theorems, with their measured instances

| theorem | statement | measured |
|---|---|---|
| `Asink_computes_chain` | the head reproduces the chain path product | **2.2e-16** at `s=8` |
| `gate_zero_not_stochastic` | `g ≡ 0` row sums to `i+1`, never `1` | witness `i=1`, row `(1,1)`, sum `2` |
| `three_corners_containment` | softmax ∪ linear ∪ path product ⊆ one family | corners distinct `4.472918 / 1.144938 / 5.335671` |
| `gate_zero_beta_zero_is_linear_attention` | the `g≡0, β=0` corner is **linear**, not softmax | row sums `1.000000` at `β=1` vs `1.312192…10.293107` at `β=0` |
| `no_prefix_scan_represents_a_zero_gate` | `exp` is never `0`; the path product is | **133,120 / 133,120 NaN** on BED-M's causal triangle |
| `no_row_stochastic_with_drive_values` | conditional impossibility — hypothesis is the content | witness `a≡1, b≡1`: `y_i = i`, stochastic row gives `1` |
| `resolvent_inverse_is_difference` | `(I−A)⁻¹` has inverse `(I−A)` — first-order, `O(nnz)` | **8.882e-16**, two planted sources |
| `first_order_cannot_delay` | arbitrary `f`, not merely affine | zero drive `0 = f(0)(0)`; impulse `1 = f(0)(0)` |
| `unit_phase_product` + `phase_path_le_one` | `\|Π a\| ≤ 1` by construction, `= 1` iff band | worst `\|a\|` = **1.0 exactly** over 2,200,000 draws |
| `six_never_reaches_the_band` | `m = exp(−softplus w)` never reaches `1` | strict for every `w`, every `θ` |
| `lru_modulus_lt_one` | LRU's `\|λ\|` is open in `(0,1)` | `1−\|λ\| = 5.0759589e-435` at `ν = −1000`, 600 digits |

### 2.2 Trivial versions refused, in-file

- `inverse_identity_is_vacuous` — depends on **no axioms**, used by nothing.
- `scan_assoc` stated for the affine monoid, not `add_assoc`.
- `delay_realizable_at_dimension_d` — a shift register *is* first-order and
  delays exactly at state dim `d+1`.
- `unit_phase_does_not_bound_the_gate` — a unit-phase gate of modulus **285.07**,
  R1's own measured divergence.
- `corners_are_distinct` — a containment whose corners coincide is decoration.

---

## 3. IDENTITY BINDS — MEASURED

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BIND                          MEASURED              BAR         SOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  oracle gates ⇒ label          5.919777e-16          1e-6        arm_smprime
    (Re part exactly 0.000000e+00; 96.78 % zero-hop fraction)
    n=512 s=64                  7.550528e-16
    n=256 s=128                 7.550528e-16

  label bind, ARM PL            6.6613381477509392e-16  1e-6      arm_pl
  label bind, ARM PHASE         9.155133597044475e-16   1e-6      arm_phase
    (bitwise identical CPU and CUDA)

  softmax corner, row sums      1.000000                exact
  β = 0 rows                    1.312192 … 10.293107
  |a| ≤ 1                       worst 1.0 exactly, 0 exceedances / 2,200,000
  band modulus                  1.000000000000        (9767/10⁴ exactly 1.0,
                                                       233 one ulp low, 0 above)
  parity = Z₂ winding           torch.equal, 0 / 4096 disagreements
  DAG resolvent, light gates    2.78e-17
  DAG resolvent, heavy gates    3.55e-15 abs / 2.16e-16 rel  (entries to 16.47)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.1 Planted negatives — every bind ships a demonstrated rejection region

| mutilation | ARM PL | ARM SMPRIME |
|---|---|---|
| drop key bias | `0.97494590151405114` | `—` |
| drop value rescale | `0.9165274652308163` | `—` |
| drop BOS sink | `1` | `—` |
| half key bias | `0.48449311856985267` | `—` |
| drop phase | — | `1.934830` |
| drop magnitude | — | `0.466267` |
| `β = 1` at the path corner | — | `1.335288` |
| **`exp_scan`** | — | **`nan`, 133,120/133,120** |

**Why this matters.** The two-branch form `softmax(q)@V₁ + λ·X@V₂` passes parity
**bitwise** for the honest gate, for Gaussian noise, and **for the label itself**.
Empty rejection region. Filed `V-24`.

---

## 4. THE DECIDING MEASUREMENT — R1

BED-M, `t* = 2`, `n = 2048`, `N = 8` seeds, CPU, threads 8.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VERDICT: NOT CROSSED.  floor₁ = 0.7071067812

  ARM PL      mean 0.829151   sd 0.253673   95 % CI [0.617075, 1.041227]
  softmax     mean 0.951767   sd 0.011824
  ĥ           0.625017        (campaign ceiling before this: 0.389)

  BIMODAL — nothing between 0.663 and 1.113
    crossed      5 / 8   NRMSE 0.634002 … 0.662021   ĥ 1.1235 … 1.1961
                         gate-R² 0.97 … 0.99         â_max 1.10 … 1.51
    NO READING   3 / 8   NRMSE 1.113403 / 1.139404 / 1.152430
                         gate-R² 0.011 / 0.627 / 0.047
                         â_max 20.31 / 49.66 / 285.07
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Two campaign firsts inside a negative verdict.** `ĥ = 0.625` exceeds the
nine-cell ceiling of `0.389`, and **no prior arm produced a single seed below the
floor at any cell — this one produced five.** `C-CAP` asks for an interval and
this interval straddles.

**Resolution statement** (`N = 8`, no TOST):

> Paired contrast (PL − softmax) `= −0.122616`, sd `0.257560`. Excludes a
> difference beyond `Δ = t(.975,7)·sd/√8 = 0.215326` NRMSE and nothing smaller.

The improvement is **52× M-10's thread-count floor** (`2.345e-3`) and **24× the
equivalence margin** — and is **not resolved**, because the three failing seeds
inflate the paired sd. Achieved power `0.232077`.

**Diagnostics.** `sign(a)` probe: trained `0.917953` vs **zero-step `0.943741`** —
negative gain. Gate-`R²`: `0.699299` trained vs `0.371839` at zero steps,
separating the populations cleanly. Corpus-alone gate-`R²` reads **`1.000000`**
because the target *is* an input channel.

**Wall clock.** `1358.32 s` total; ARM PL `91.63 s`/150 steps at `n=2048`,
softmax `71.32 s`. The `5.8×` gap against the fitted law was later **refuted** —
see §6.

---

## 5. BEDS AND INSTRUMENTS

### 5.1 BED-K — the first corpus whose label carries a delayed cause

Every corpus before R11 was Markov-generated, so no label had ever contained a
delayed cause.

| test | measured |
|---|---|
| delay bed scan-blind | `R² = −0.000170` using the label's own true previous value |
| the control that validates it | **same fitting code** recovers AR(1) at `α = 0.8000, β = 1.0000, R² = 1.000000` |
| 30-seed sweep | `[−0.0027, 0.0028]` |
| delay bed attention-reachable | `8.67e-19` against a `1e-12` bar |
| **power-law bed scan-blind?** | **NO — `R² = 0.604` at `H = 0.75`, `0.755` at `H = 0.9`** |
| Jacobian oracle | delay `2.88e-11`, power-law `4.88e-10`, both `≤ 1e-9` |
| Hurst (DFA, not R/S) | white noise `≈0.478` · AR(0.5) `≈0.486` · R/S's biased reading is `0.75` |
| Hurst recovery bias | `0.812 → 0.818` at `H_true = 0.75` — the FARIMA correction **did not shrink it**, reported as such |

**#12 is an exact-identity result, not an approximation bound.** A recurrence that
cannot reproduce a kernel *exactly* may still fit 60% of its variance — and does.

### 5.2 BED-1 — multi-basin, committor labels

28 tests. `T* = ΔΔE‡/ln m = 0.62133493455961186`.

| | `0.9 T*` | `1.1 T*` |
|---|---|---|
| low-barrier rate | `1.672502e-01` | `2.315116e-01` |
| `m`-fold high-barrier bundle | `1.398632e-01` | `2.679881e-01` |
| barrier label | `lo` | `lo` |
| committor label | `lo` | **`hi`** |
| | **AGREE** | **DISAGREE** |

Crossover bisected at `0.62133493455961175` against the closed form —
**`|diff| = 1.110e-16`**. A **third independent route** from the dynamics:
empirical `lo:hi = 0.86679 < 1` at `1.1 T*` — walkers cross the five-fold
*higher* barrier more often, agreeing with the committor ratio to `0.34%`.

**Committor harmonic:** `max|Lq| = 0.000000e+00` default, `1.040834e-17` jittered;
must-fire perturbation drives it to `1.000000e-06` — **11 orders**.

**CK test:** `τ=1 → 0.22702` (fails), `τ=64 → 0.00400` (passes), factor 57. Both
vacuity ends guarded *at the passing lag*: `‖T̂(64)−I‖ = 0.9482`,
`‖T̂(128)−Π‖ = 0.1280`.

**Pesin deficit:** `h = 0.081275` nats. Generating partition **`0.000233`**; wrong
guards `0.029889 / 0.030616 / 0.029915` — factor 128. **The `q=½` guard partition's
own deficit is `0.009654`, 11.9% of `h` — NOT near zero, so the contract's guards
are not a generating partition.** Reported, not buried.

**Conservation census: 7 conserve, 2 do not**, printed with mechanism. Non-conserving:
guard-crossing balance (21 net on 65,481 events) and trap-channel share
(`0.11005` vs `0.05837`, off by `1.885×`, because a channel with a metastable
interior recrosses its own guard).

**Morse census:** `m₀=3, m₁=8, χ = −5 = m₀−m₁, b₁=6`. Must-fire moves the
classifier to `(2,7)` and leaves `χ` unchanged.

**Five mutants killed**, on the stated grounds that *13/13 green on a first
implementation run isn't evidence*.

### 5.3 X₃₅′ — hidden-cause detection

| instrument | status | measured |
|---|---|---|
| Shewhart onset | **NOT VOID at oracle scope** | onset `36` at `sd=0` and `sd=0.05`, true `36`; 500 seeds → `{0:421, 1:74, 2:5}`, **0.9900 within ±1** vs a pre-registered `0.95` |
| flatness gate rejection region | **demonstrated** | exact model FAR `0.0120`; lag-64 truncated **`0.2667` (27×α)**; lag-8 **`1.0000` (100×α)** |
| FAR scored out-of-sample | yes | calibrated `[0,2000)`, scored `[1e6, 1e6+2000)`; in-sample `0.0095` printed beside, not substituted |
| exact source solve | `8.882e-16` | two planted sources — the delta's cited `8.9e-16` |
| exact inverse vs adjoint | **RETIRED to noiseless** | crossover `sd = 0.15`; by `sd=0.50` it is 9 vs 266 with 67 exact misses against 5 |
| Kramers–Kronig | **SHIPS** | planted anticipating kernel `0.98994949366116647` vs null max `1.3670e-15` over 262 causal kernels — **margin `7.24e14`** |
| KK closed form | `√2·‖anticausal‖₂` | independent route agrees `1.11e-16`; **no threshold, no tuning knob** |
| Cramér–Rao floor | **DOES NOT EXIST** | discrete onset; smoothing makes the bound a property of the smoothing (`2462×`, then `1.08e7×`) |
| Ziv–Zakai (ships instead) | calibrated **exactly** | `0.00e+00` difference at `n = 32/64/128`; Bayes-optimal tracks it across a **4096× SNR span**, ratio `[0.998, 1.329]` |

**The KK diagnosis ran all four candidate mechanisms** rather than hypothesising:
origin-forgetting `1.657e-16` (zero on every input), residual on `|H|`
`3.269e-16` (zero by construction), no zero-padding `0.000000e+00`
(conditionally), and `1/M` normalization **eliminated** — it shrinks as `1/√M`, so
an exact zero would need `M > 1.3e7`.

### 5.4 X₃₇ — topological certificates

| certificate | status | measured |
|---|---|---|
| `Z` winding | ships, **with a second guard** | correct on `k ∈ {−3,−1,0,1,2,5,15}`, worst departure `4.4e-16`; refuses aliased input where the unguarded path returns a **plausible but wrong `−2`** |
| — the defect it found in itself | **63 turns in 64 samples has a wrapped step of `0.031π` — the healthiest reading in the file — and returns `−1`.** Refinement check added | |
| persistent `β₁` | ships | RPS one bar `(0.045414, 0.551497)`, persistence `0.506083`; coordination control **no finite `H₁` bar at any radius** |
| stability | `2ε` bound holds | `1.26e-3 / 1.07e-2 / 5.48e-2` against `2e-3 / 2e-2 / 1e-1` |
| Euler–Poincaré | **AGREES, and half of it cannot fail** | `Σ(−1)^k β_k = 1−6 = −5` = BED-1's `m₀−m₁ = −5`. **On a 1-complex `β₀−β₁ = V−E` identically** |
| — and it fires falsely where it can | on a Hopf normal form with a **complete** census, `Σι = 1 = χ(D²)` while the Rips carrier reads `Σ(−1)^k β_k = 0` | |
| Kuramoto order parameter | both halves | locking matches `exp(−σ²/2)` at `0.999800/0.998743/0.980110`; `N=4096` equispaced → **`5.55e-17`**; uniform 8 seeds → mean `0.012916` vs Rayleigh `0.013847` |

---

## 6. SYSTEMS — THE DEVICE CERTIFICATE

**Certified device: CUDA, RTX 4060 Laptop, sm_89.**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  wall-clock law    s/step = exp(−11.9670) · n^0.9734       R² 0.999384
  40-iteration plan 68.35 GPU-h  (1.07 at the 150-step floor)
                    815.88 CPU-h (12.75 at the floor)
  the decision      11.94×  ·  29.0× at the pessimistic end
                    (the contract's cited 13.6× is the optimistic reading)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.1 The `5.8×` gap does not reproduce

R1's softmax cell — same arm, corpus, thread pin, steps, statistic — reads
**`12.3 – 13.6 s`** against R1's filed **`71.32 s`**.

**Eliminated by measurement:** thread count (`0.99–1.26`), data source (`1.01×`),
the arm (`1.06–1.11×` vs R1's filed `1.285`), the statistic (`1.012×`), thermal
(**wrong sign** — R1's own block got *faster*), dtype and shape (identical by
construction).

**What survives:** an **i7-14700HX, 8 P-cores + 12 E-cores**, where
`set_num_threads(8)` names a count and **never a placement**. E-cores alone
`1.44×`; host contention alone `2.55×`; **their cross spans `2.0×–4.0×` across six
measurements of one configuration.**

**Dated corroboration, and it is this session's own doing:** a `lake build` wrote
`V15Source.olean` at `17:43:35`, inside R1's softmax seed-3 window, and the four
cells after it are **13.1% cheaper** than the four before.

Not claimed: the full factor. **41–83% in log terms; the residual is unexplained.**

**The consequence:** under identical pressure, **CPU spread `2.55×`, CUDA spread
`1.05×`**.

### 6.2 Memory — the model holds, and breaks where it is new

| arm | `measured / predicted` |
|---|---|
| softmax | **`0.950`** across a 16× span |
| ARM PL | **`0.950`** |
| **ARM PHASE (complex)** | **`1.842` — the module UNDER-predicts** |

`C_OPERATOR` for the complex arm measures **`7.50` at 8 B/element** — `4.29×`
softmax's, not the `2×` a complex-is-two-floats argument gives. `C_RESIDUAL`
**NOT IDENTIFIED** at `s=64`.

**R2 at `n = 32768` DOES NOT FIT:** `7.893 GiB` against `6.939` free, `7.996`
total. Not attempted. Does not fit on CPU either. Cap `22,106` at 20%, `27,633` at
zero; **last usable power of two `16,384`** — where it is also *cheaper*, `5.29 h`
vs `10.38 h`. Four contract cells affected.

### 6.3 Determinism

`cumsum` has **no deterministic CUDA kernel** in torch 2.5.1 and is the whole of
the scan arms. The hazard was measured to a boundary: **bitwise at R1′'s shape,
bitwise at R2's exact `[32768, 64]`, bitwise at 16.8M elements, bitwise at
reduction 4,096 — drifting only at reduction `1e6`.** *The hazard follows
reduction length, not launch size, and the arms reduce over 64.*

**`cumprod` has a deterministic CUDA kernel.** `arm_smprime` therefore runs under
`use_deterministic_algorithms(True)` where `arm_phase` raises.

TF32 asserted off, measured: enabling it moves the operator by **`3.184e-04`**
relative against `3.237e-07` off.

### 6.4 The bar re-certified across devices

| clause | worst δ | tolerance | δ/tol |
|---|---|---|---|
| `predict_the_mean` | `8.580e-08` | `1e-6` | **8.6%**, 11.7× headroom |
| `flipper_dependence` | `1.053e-07` | `0.05` | `2.1e-06` |
| `payload_only` | `9.130e-08` | margin `6.41e-02` | `1.4e-06` |
| `trained_two_feature` | `3.689e-08` | margin `9.86e-01` | `3.7e-08` |
| `oracle` | `0` exact both | `1e-6` | `0` |

Tolerances taken from `bar_verdict`'s **own body, text predating the node** — not
refitted. Behaviour preservation verified by loading the previous revision as a
second module: **all 30 doubles bit-identical in IEEE-754 hex.**

0-step gate at R1′'s shapes: **32 cells, all PASS both devices**, max
`|cpu−cuda| = 1.679e-07`, smallest margin `9.7039e-04` — **5,780× the gap**.

---

## 7. THE FAILURE TAXONOMY — 65 MECHANISMS

`MISTAKES.md`. Each entry: a type name, a real instance with `file:line`, a
measured number, and a check.

**Filed this session (11):**

| id | mechanism |
|---|---|
| `M-17` | a census that classified the **correction record** as the defect |
| `M-18` | a kill-diagnostic whose value the corpus fixes — **with a correction against its own author** |
| `M-19` | a dynamical invariant estimated on a float64 orbit that has already collapsed |
| `M-20` | a pre-registration that predicts **both** outcomes, in two sections that never met |
| `M-21` | a diagnostic prescribed by its **statistic** instead of by what it must distinguish |
| `P-10` | a source's intro cited as its theorem |
| `P-11` | a contract citing its **own `[M]`-tagged** theorem as settled |
| `V-23` | a plural claim whose own central member is the counterexample |
| `V-24` | an identity bind whose **rejection region is empty** |
| `V-25` | a theorem whose hypothesis **no draw in the corpus satisfies** |
| `D-7` | a prediction filed without its counter, in a document whose errors have a sign |

### 7.1 The three laws each was paid for

**L-DOM** — every theorem gating a run ships a domain census.

> Lean #2's hypothesis is `∀k, 0 < a k`. BED-M's support is `{−1, 0, +1}`.
> **1 of 3 values; 1.58% of entries; and ZERO of 2,048 sequences — structurally**,
> because `negation_scope.py:429` zeroes the first `head+1` positions of every
> row. *The quantified hypothesis is unsatisfiable on any draw the builder can
> produce.*

**L-SIGN** — a counter-prediction beside every prediction, calibration carried
across rounds.

**L-DIAG** — a contract may prescribe what a diagnostic must *distinguish*, never
which statistic does it.

### 7.2 The domain census condemned three more theorems

| theorem | BED-M | BED-K |
|---|---|---|
| #2 original | 1 of 3 | **EMPTY** |
| #2 re-stated | **3 of 3** | still empty — no gate sequence to quantify over |
| **#6 `bounded_gates_stable`** | **0 of 3 — EMPTY** | EMPTY |
| #12 | EMPTY | partial on (a); `d=0` necessarily out |
| #16 | total overlap, **NULL content** |

**#6 being empty is not bookkeeping:** an arm with #6-parametrized gates **cannot
be set to BED-M's oracle gates at all**, so the identity bind is not a test that
family can be given.

**#16 exposed a hole in L-DOM one iteration after it was filed:** a
hypothesis-free theorem passes an overlap test **vacuously**. A census needs two
columns — hypothesis overlap *and* what the theorem constrains on the drawn value.

---

## 8. THE CALIBRATION COLUMN

| round | checked | adverse | signs | one-sided test |
|---|---|---|---|---|
| **R11** | **9** | **9** | 7 `+`, 1 `−`, 1 unsigned | **`7/8`, `p = 0.0352`** |

**The summary that opened this column was itself wrong three ways**, and the
correction *strengthens* the finding:

1. It said "six wrong of nine checked". **All nine rows carry adverse verdicts.**
   Three were dropped in transcription — all three discussed elsewhere in the same
   document.
2. It said all six were optimistic. **One is pessimistic** — a completed sizing
   repair reported as outstanding *understates* the project. One is unsigned.
3. It had **no denominator discipline** — no confirmations counted, so `9/9` is a
   count, not a rate. The one sub-census with a real denominator reads
   **4 findings of 11 checked = 36.4%**.

**On the corrected nine: `p = 0.0352` (significant). On the original six:
`p = 0.1094` (not).** The headline was right about the direction and wrong about
its evidence.

**The discount rule** — `D-CALIB-1` through `-5`:

- **The counter is the point estimate**, not the tail.
- **A bare prediction is blocked, not discounted.**
- **Sign, never size.** Wilson 95% is `[0.5291, 0.9776]` — an ordering, not a
  scaling. *A multiplier invented at this `N` would be a fresh `P-1`.*
- Schedule the cheapest refutation of the optimistic half first.
- Append the row whether or not it flatters.

---

## 9. PRIOR ART — WHAT IS OCCUPIED

**14 of 14 lineages reached `[V-eq]`** — equation transcribed, hypotheses as the
source states them, and a numeric instance run, **each probe carrying an `O(1)`
mis-transcription control** so a `1e-16` residual is evidence rather than
decoration.

| component | occupant | verdict |
|---|---|---|
| decay masks | SSD/Mamba-2, GLA, RetNet | occupied |
| phase gates, `m ≡ 1` face | uRNN (Arjovsky 2016), RoPE | occupied |
| polar factorization | LRU (Orvieto 2023), `Λ = diag(exp(−ν+iθ))` | **verbatim** |
| **closed magnitude** | **S4D ReLU variant** — `\|Ā\| = 1.0` exactly on **32.93%** of a standard sample, published 2022 | **occupied** |
| `m = 0` attainable | modReLU, **14.70%** of samples | occupied |
| hidden-cause composition | **Basseville & Nikiforov 1993 §7.2.4**, equation for equation | **occupied since 1993** |
| — learned-model form | arXiv:2604.25655 (28 Apr 2026), **Thm 3.1 is the must-fire as a theorem** | occupied |
| fractional head | **VORT** (arXiv:2605.08966), verified from arXiv API | occupied |

### 9.1 Two near-misses worth recording

**The fetch summarizer would have confirmed a false claim.** Extracting LRU §3.3
from the rendered PDF with `pypdf` returns text in which the Type-1 encoding maps
`−` → `\x00` and `∞` → `1`, turning *"`|λ|=1` achieved at `ν = −∞`"* into its
opposite. **The single sentence that decides that node is the one the extractor
corrupts, and it corrupts it into the claim under test.** All ML papers were
therefore read from arXiv LaTeX source.

**A published ablation answers a counter-prediction.** Goel/Gu/Donahue/Ré
arXiv:2202.09729 §3.1 documents S4 matrices *"generally became non-Hurwitz after
training"* — the same instrument R1 built — and its ablation reads **unstable
`1.420` NLL vs stable `1.419`**: stability cost `0.001`, in the improving
direction. Nothing reports a magnitude of `285`.

---

## 10. TEST STATE

```
tests/loop        15 failed / 523 passed    the same 15 by name
tests/arm_smprime 35 passed CPU  ·  35 passed CUDA, same ids
tests/arm_phase   30 passed CPU  ·  30 passed CUDA
tests/arm_pl      17 passed
tests/beds        28 passed      (BED-K 15 + BED-1 13)
tests/certs       36 passed      (8 mutants killed)
tests/x35         14 passed
tests/x35p        29 passed
tests/mars_v15    21 passed / 4 skipped   every skip must-fired
```

**The 15 standing failures are each a bound finding with a stated route, not
breakage.** Membership was stable across the last five snapshots; the *pass* count
has moved five times (`501 → 515 → 517 → 518 → 522 → 523`) because two repo tests
parametrize over every module and gain a case whenever one is added. **The failure
set is the invariant; the pass count is not.**

---

## 11. WHAT IS OWED

| debt | why it blocks |
|---|---|
| **D-APPROX** — Lean #17 approximation bound | #12 is an *exact-identity* result. The power-law bed is not scan-blind at `R² = 0.604` and that does not contradict it. **Every "X cannot represent Y" claim needs a bound before it reads as "X cannot fit Y."** The largest unclosed gap in the round's own logic |
| **D-R3** — two inverse registrations | PART III and PART IV predict opposite outcomes for the same bed. The author picks one; the cell cannot run first |
| **R2's `n`** | `32768` does not fit; `16,384` is the last power of two |
| **`[0,1]` vs `[0,1)`** | Three nodes measured the closed endpoint as the defect. A fourth refuted that: **the endpoint is not the failure, `log` of it is** — the direct-product route completes 40 steps at 55.58% zeros. And `[0,1)` would cost reachability of `a = ±1`, **two-thirds of BED-M's support** — the same defect as #6, one endpoint over. **Filed two-sided for the author** |
| **the scan skyline** | correctly refused in R11 as a new construction; legal now as the native control |
| **HF package** | scheduled, unbuilt. No trained checkpoint ships |

---

## 12. THE ONE CLAIM SENTENCE THE TABLES PERMIT

> **A single causal softmax head, carrying a prefix-scan in its key logit and a
> value-zero sink, is bitwise standard attention at its identity setting and
> reproduces an exact path-product label on the same head — to `5.9e-16` on the
> corpus's real support, with planted mutilations failing at `O(1)`.**

It claims **no capability advantage over softmax**. None has been measured.

It claims parity at *both auxiliary heads zero* — *the modification enters only
through the key logit, additively, and vanishes at zero* — **not** "bitwise
standard attention" unqualified, which is what the refuted clause said.

It claims **no component novelty**. All lineages reached `[V-eq]` and are
occupied.

---

## 13. DISTANCE TO THE NORTH STAR

> *Attention that is EQUAL to self-attention on its own ground, built FROM softmax
> and AdamW, and capable on ground they cannot occupy — predicting the NEXT STATE
> toward equilibrium, not the next token.*

**Clause 1 — equal on its own ground.** Went from *refuted* to *bitwise*, on an
amended operator, with the corners proved distinct.

**Clause 2 — built FROM softmax.** Went from *false* — the original hop was not a
softmax at all — to *literal*, to *instantiated*: a softmax row at every `θ`.

**Clause 3 — capable on ground they cannot occupy.** **Unmoved.** Five of eight
seeds crossed a floor no arm had ever crossed and the interval straddles it. The
binding constraint is where the algebra always said it was — **multiplicative
composition along paths** — now bounded by measurement: order-0 features reach
`R² = 0.01–0.02` against the scored label, against `1.000000` on the gate.

**The honest summary.** Two rounds have made the next round's measurements mean
something and taken one. The apparatus is sound, the claim is unmade, and the
error that would most have mattered — a capability number believed for the wrong
reason — did not happen, because every instrument that could have produced one
carries a way to fail that it is required to report.
