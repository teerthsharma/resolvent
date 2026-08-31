# V15 BED-1 — the committor-labelled multi-basin bed

`CEQ_V15_CONTRACT.md` PART III (**BED-1**), PART IV (**R5**), contract item #14
`committor_eq_harmonic [S]`. Node: CAMERON-2, CEQ v15.1 composition round.

Files added: `ceq/beds/bed_1.py`, `tests/beds/test_bed_1.py`. One line added to
`ceq/beds/__init__.py` (`__all__` and the module docstring's third sentence).
Nothing else in the tree was touched; `ceq/beds/bed_k.py` and
`tests/beds/test_bed_k.py` are byte-identical to their pre-round state.

`tests/beds`: **28 passed** (13 BED-1, 15 BED-K, all BED-K tests still green).

---

## 1. What BED-1 is

An 11-node energy graph, three basins and eight saddles, whose label is the
splitting probability `q` and never the barrier height.

```
   A(0.0) ---- S_lo(1.0) ---------------------- B(0.0)      the low channel
   A      ---- S_hi0..4(2.0) ------------------ B           m = 5 parallel high saddles
   A      ---- S_ac(1.5) -- C(0.9) -- S_cb(1.5) B           via a metastable intermediate
```

* **Rates.** Metropolis, `k_ij = exp(-(V_j - V_i)_+ / T)`, reversible with
  respect to `w_i = exp(-V_i / T)`. Discrete-time chain by uniformization,
  `P = K / d` with `d = max_i sum_j k_ij`.
* **Label.** `q` solves `(Lq)_i = 0` on the interior, `q = 0` on `A`, `q = 1`
  on `B`. Dense float64 solve of the 9×9 interior system.
* **Guards.** The `q = 1/2` level set — computed from the committor before any
  trajectory exists. On the default landscape that set is
  `{S_lo, S_hi0..4, C}`: six saddles and the intermediate basin.
* **Itinerary.** A maximal run of `q = 1/2` states entered from one side of the
  isocommittor surface and left on the other, labelled by the channel of the
  guard node it passed through. `direct = 0` at every temperature measured —
  no crossing bypasses a guard.
* **Channels.** `lo` (barrier 1.0, one saddle), `hi` (barrier 2.0, five
  saddles), `trap` (barrier 1.5, through `C`).

`ΔΔE‡ = E_hi − E_lo = 1.0`, `m = 5`, so
`T* = ΔΔE‡ / ln m = 0.62133493455961186`.

**Why `C` is there.** It is a genuine transition-state basin sitting exactly on
the `q = 1/2` surface, and it is what gives the CK battery a lumping with real
memory: a walker that entered the TS region through a saddle leaves in one step,
one that entered at `C` stays for `~e^{0.6/T}` steps. Without it the committor
macrostates are Markovian at every lag and the CK test measures nothing.

---

## 2. RED

`tests/beds/test_bed_1.py` was authored and run before `ceq/beds/bed_1.py`
existed.

```
$ python -m pytest tests/beds/test_bed_1.py -q
=================================== ERRORS ====================================
__________________ ERROR collecting tests/beds/test_bed_1.py __________________
ImportError while importing test module 'C:\Users\seal\Desktop\New folder (32)\tests\beds\test_bed_1.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\beds\test_bed_1.py:45: in <module>
    from ceq.beds import bed_1
E   ImportError: cannot import name 'bed_1' from 'ceq.beds' (C:\Users\seal\Desktop\New folder (32)\ceq\beds\__init__.py)
=========================== short test summary info ===========================
ERROR tests/beds/test_bed_1.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.47s
```

## 3. GREEN

```
$ python -m pytest tests/beds -v
collected 28 items

tests/beds/test_bed_1.py::test_committor_is_harmonic_in_the_interior_and_the_check_fires[0.0] PASSED
tests/beds/test_bed_1.py::test_committor_is_harmonic_in_the_interior_and_the_check_fires[0.05] PASSED
tests/beds/test_bed_1.py::test_committor_matches_its_closed_form_on_the_symmetric_landscape PASSED
tests/beds/test_bed_1.py::test_crossover_temperature_is_delta_delta_over_ln_m PASSED
tests/beds/test_bed_1.py::test_barrier_and_committor_labels_agree_below_Tstar_and_disagree_above PASSED
tests/beds/test_bed_1.py::test_ck_passes_at_the_right_lag_and_fails_at_a_wrong_one PASSED
tests/beds/test_bed_1.py::test_pesin_deficit_near_zero_at_generating_partition_and_large_at_wrong_guards PASSED
tests/beds/test_bed_1.py::test_guards_are_the_q_half_level_set_and_were_not_chosen_by_deficit_argmin PASSED
tests/beds/test_bed_1.py::test_morse_census_closes_on_the_euler_characteristic_and_moves_when_V_moves PASSED
tests/beds/test_bed_1.py::test_conservation_census_names_a_non_conserving_carrier PASSED
tests/beds/test_bed_1.py::test_guard_itinerary_channel_shares_reproduce_the_committor_flux_ratio[0.9-lo] PASSED
tests/beds/test_bed_1.py::test_guard_itinerary_channel_shares_reproduce_the_committor_flux_ratio[1.1-hi] PASSED
tests/beds/test_bed_1.py::test_seeded_reproducibility_bitwise PASSED
tests/beds/test_bed_k.py  ... 15 items ... PASSED

============================= 28 passed in 12.15s =============================
```

### 3b. The suite was mutation-checked, because 13/13 on the first implementation run is not evidence

Five mutants introduced one at a time into `ceq/beds/bed_1.py`, suite re-run,
module restored:

| mutant | tests killed |
|---|---|
| M1 `label_by_committor := label_by_barrier` | `test_barrier_and_committor_labels_agree_below_Tstar_and_disagree_above` |
| M2 `harmonic_residual := 0.0` | both parametrizations of `test_committor_is_harmonic_...` |
| M3 `guards := every node` | `test_guards_are_the_q_half_level_set_...` |
| M4 `committor := 1/2 on the whole interior` | 4 tests, incl. closed-form and conservation |
| M5 census returns before its non-conserving rows | `test_conservation_census_names_a_non_conserving_carrier` |

No mutant survived. The suite is not a set of tautologies over its own outputs.

---

## 4. (a) The committor: harmonic check and its must-fire

Contract item #14. `max_i |(Lq)_i|` over the interior, and the same quantity
after one interior node (`C`) is moved by `1e-6`.

| landscape | interior `q` | residual | must-fire residual | ratio |
|---|---|---|---|---|
| `jitter = 0.0` (default) | `0, ¼, ½, ¾, 1` | **0.000000e+00** | **1.000000e-06** | ∞ |
| `jitter = 0.05`, seed 11 | `q(C) = 0.3356422966`, `q(S_ac) = 0.1678211483`, `q(S_cb) = 0.6678211483` | **1.040834e-17** | **1.000000e-06** | 9.608e+10 |

**Yes, and yes.** The check passes at machine precision and the must-fire fires
by 11 orders of magnitude. Test threshold for the pass is `1e-13` (11 orders
above the measured jittered residual) and `1e-9` for the fire.

Two landscapes rather than one: at `jitter = 0` the committor is dyadic and the
residual is *exactly* zero, which is the strongest possible pass but also a
special case; the jittered landscape has a non-dyadic `q` and exercises the
general solve. `build()` refuses a jitter that reorders any edge's endpoints
rather than silently emitting a landscape that is no longer this bed.

**A stronger check than the residual, also shipped.** The default landscape's
committor is analytically known and *temperature-independent*: under Metropolis
rates every downhill rate is exactly 1, so a barrier top flanked by two deeper
basins has `q = (q_left + q_right)/2` for any depths and any `T`. That gives
`q(S_lo) = q(S_hi_k) = q(C) = 1/2`, `q(S_ac) = 1/4`, `q(S_cb) = 3/4`. The 9×9
`np.linalg.solve` reproduces those **bitwise** at `T = 0.25, 0.3, 0.9T*, 1.1T*`.
Checking a solve against a closed form ranges over a different set than checking
it against its own residual (MISTAKES.md V-3).

---

## 5. (b) The strike: barrier-height vs committor labelling

All figures below come out of the **committor solve** (`2 · J_channel`, the
factor 2 being the channel's two edges in series — equivalently the bare escape
rate, of which the committor keeps half at `q = 1/2`), not from the closed form
they are compared against.

| T | `2J_lo` | `2J_hi` (bundle) | `2J_trap` | barrier label | committor label | |
|---|---|---|---|---|---|---|
| `0.90 T*` = 0.5592014411 | **1.672502e-01** | **1.398632e-01** | 3.419952e-02 | `lo` | `lo` | **AGREE** |
| `1.00 T*` = 0.6213349346 | 2.000000e-01 | 2.000000e-01 | 4.472136e-02 | `lo` | tie | crossover |
| `1.10 T*` = 0.6834684280 | **2.315116e-01** | **2.679881e-01** | 5.569665e-02 | `lo` | `hi` | **DISAGREE** |

Closed-form Arrhenius at the same temperatures: `1.672502e-01 / 1.398632e-01`
and `2.315116e-01 / 2.679881e-01` — identical to the committor readings to
printed precision, as they must be, since `J_hi / J_lo = m·exp(-ΔΔE‡/T)`
exactly on this landscape.

**Both halves are asserted**: below `T*` the two labelling schemes must agree,
above it they must differ. The barrier label is additionally checked to be
*constant* across `T ∈ {0.2, 0.4, 0.9T*, 1.1T*, 1.0, 2.0}` — that constancy is
its defect, and pinning it prevents "the labels disagree" from being an artifact
of a wobbling barrier labeller.

**The crossover temperature, located from the committor rather than assumed.**
200 bisection steps on `[0.3, 1.5]` of `J_hi(T) − J_lo(T)`:

```
bisected crossover (from committor flux) = 0.62133493455961175
closed form  T* = DDE / ln m             = 0.62133493455961186
|difference|                             = 1.110e-16
```

Same order as the contract's `[reproduced to 6.4e-16]`.

At exactly `T*` the two fluxes are equal to within one ULP and the reported
label is decided by float rounding; nothing is asserted there.

---

## 6. (c) CK test — `T̂(nτ) ≈ T̂(τ)ⁿ`

Committor macrostates `{q < ½, q = ½, q > ½}`, i.e. `{A-side, transition state,
B-side}`, at `T = 0.25`, 2000 walkers × 6000 steps (1.2e7 samples), seed 0,
`n = 2`. Chain implied timescales at that temperature: **93.75 / 22.24 / 0.33**
steps.

| τ | CK error | `‖T̂(τ) − I‖∞` | `‖T̂(2τ) − Π‖∞` |
|---|---|---|---|
| **1** (wrong lag) | **0.22702** | 0.4778 | 0.4949 |
| 2 | 0.20367 | 0.4949 | 0.4842 |
| 4 | 0.16923 | 0.5355 | 0.4635 |
| 8 | 0.11936 | 0.6083 | 0.4247 |
| 16 | 0.05840 | 0.7186 | 0.3571 |
| 32 | 0.01253 | 0.8503 | 0.2528 |
| **64** (right lag) | **0.00400** | 0.9482 | 0.1280 |
| 128 | 0.00116 | 0.9755 | 0.0342 |

**Fails at τ = 1 (0.22702), passes at τ = 64 (0.00400)** — a factor of 57.
Thresholds `> 0.10` and `< 0.02` sit between them with 2× and 5× of margin.

**Non-vacuity, at both ends.** A CK test passes for free at `τ → 0` (nothing has
happened, `T̂ ≈ I`) and again at `τ → ∞` (everything has relaxed, `T̂ ≈ Π`).
Both degeneracies are checked *at the passing lag*: `‖T̂(64) − I‖ = 0.9482 > 0.5`
and `‖T̂(128) − Π‖ = 0.1280 > 0.05`. The pass at τ = 64 is neither "nothing
happened" nor "everything relaxed". Note the τ = 128 row's `‖T̂ − Π‖ = 0.0342`,
which is *why* the passing lag was set at 64 and not higher.

Mechanism of the failure at short lag: the TS macrostate contains both saddles
(residence 1 step) and the intermediate basin `C` (residence ≈ 22 steps). The
lumping therefore carries memory that decays on `C`'s internal timescale, and
`τ = 64` is past that decay while still well inside the 93.75-step A↔B
interconversion.

---

## 7. (d) Pesin deficit, and the arithmetic

### The arithmetic, stated (MISTAKES.md M-19)

**BED-1 iterates no real-valued map, so no float64 horizon applies, and this is
a claim rather than an omission.** The dynamics are a reversible Markov jump
chain on 11 integer states. A trajectory is an `int8` array produced by
comparing PRNG uniforms against a fixed cumulative row of `P` — the state is an
integer at every step, no bit of an initial condition is lost per step, and
there is no `mantissa_bits / log2(stretching rate)` quantity to exceed. The
float64 arithmetic in this bed is: the rate matrix, the 9×9 committor solve, and
the entropies. None of those is obtained by iteration.

`λ̂` is replaced by the quantity that plays its role for a stochastic chain: the
Kolmogorov–Sinai entropy rate `h = −Σ_i π_i Σ_j P_ij ln P_ij`, computed **in
closed form** from `P` and `π`. It is not estimated from an orbit at all. The KS
theorem gives `h_sym(α) ≤ h` for every partition `α` with equality at a
generating one, and for a Markov chain the state partition is generating — so
`h − h_sym` is a non-negative deficit with the same admissibility semantics as
`λ̂ − h_sym`. Any future BED-1 variant driven by a chaotic map inherits M-19 in
full and would need the exact-rational treatment `scripts/v15_n1_probes/p08_pesin_symbolic.py`
uses.

### The readings

`T = 0.25`, same trajectory as the CK battery, exact `h = 0.081275` nats/step.

| partition | `L` | `h_sym` | deficit | deficit / `h` |
|---|---|---|---|---|
| **state partition (GENERATING)** | 3 | 0.081042 | **0.000233** | 0.0029 |
| `q = ½` guard partition, 3 symbols | 8 | 0.071621 | 0.009654 | 0.1188 |
| binary cut `q ≥ 0.50` | 10 | 0.051386 | **0.029889** | 0.3678 |
| misplaced cut `q ≥ 0.25` | 10 | 0.050660 | **0.030616** | 0.3767 |
| misplaced cut `q ≥ 0.75` | 10 | 0.051360 | **0.029915** | 0.3681 |
| barrier-height cut `V ≥ E_lo` | 10 | 0.065231 | 0.016044 | 0.1974 |

**0.000233 at the generating partition against 0.0299 / 0.0306 / 0.0299 at the
wrong guards** — a factor of 128, and normalised by `h` that is
`0.0029` against `0.368 / 0.377 / 0.368`, the same shape as the contract's
`0.0003 vs 0.139/0.208/0.223`. Thresholds: `< 0.002` for the near-zero half
(8.6× above the measurement, 4.8× below the smallest coarse reading, so the two
halves cannot swap) and `> 0.005` plus a `10×` separation for the large half.

**An honest reading the contract needs.** The `q = ½` guard partition's own
deficit is **0.009654 (11.9 % of `h`)** — not near zero. The guard partition is
*not* generating on this bed, which under R5's `(K̂, Pesin-deficit)` policy is a
distribution-predict verdict rather than a point-predict one. That is reported
rather than buried.

### Guards were **not** selected by an argmin over the deficit

**Explicit statement: no guard in this bed is chosen by minimising, ranking, or
searching over the Pesin deficit.** `guards` is the `q = 1/2` level set of the
committor, computed inside `build()` from the harmonic solve, before any
trajectory exists and without reference to any entropy. No function in
`ceq/beds/bed_1.py` takes a deficit as an input to guard selection;
`pesin_deficit` is offered only as the near-zero admissibility test Bollt et al.
(2001) support.

That claim is made checkable rather than left as a disclaimer. Deficits at three
binary cuts whose misplacements `|θ − ½|` are `0.25, 0.00, 0.25`:

```
theta = 0.25   deficit = 0.030616      misplacement 0.25
theta = 0.50   deficit = 0.029889      misplacement 0.00
theta = 0.75   deficit = 0.029915      misplacement 0.25
```

A 2.4 % spread across a 0.25 swing in misplacement, and **not ordered by it**:
the two *equally* misplaced cuts differ from each other (0.000701) by more than
one of them differs from the correct cut (0.000026). An argmin over these would
be selecting noise — Bollt's non-monotonicity, observed on this bed rather than
cited. The test asserts the spread is small relative to the values, i.e. it
asserts the *premise that makes an argmin unsound*, not a ranking.

---

## 8. (e) Conservation census — including what does not conserve

`T = 0.25`, with the 1.2e7-sample trajectory. Every carrier the bed has, with
its value and tolerance.

| | carrier | value | tol |
|---|---|---|---|
| CONSERVES | transition rows sum to 1 | 1.110223e-16 | 1e-12 |
| CONSERVES | stationary distribution `πP = π` | 1.110223e-16 | 1e-12 |
| CONSERVES | detailed balance `π_i P_ij = π_j P_ji` | 1.355253e-20 | 1e-12 |
| CONSERVES | committor harmonicity `(Lq)_int = 0` | 0.000000e+00 | 1e-12 |
| CONSERVES | reactive flux divergence-free on interior | 0.000000e+00 | 1e-12 |
| CONSERVES | reactive flux `out(A) = in(B)` | 1.734723e-18 | 1e-12 |
| CONSERVES | itinerary `lo:hi` ratio vs committor flux ratio | 8.113675e-03 | 2e-2 |
| **DOES NOT** | **guard-crossing balance `#(A→B) = #(B→A)`** | **2.100000e+01** | 0 |
| **DOES NOT** | **itinerary share vs reactive flux, TRAP channel** | **8.853510e-01** | 5e-2 |

**The two that do not conserve, printed rather than omitted (MISTAKES.md V-23):**

1. **Guard-crossing balance.** Forward and backward guard traversals are equal
   only *in expectation*. Measured: 21 net on 65,481 events, i.e. **0.08
   sqrt-units**. At `0.9T*` and `1.1T*` the same row reads 42 on 851,126
   (0.05 sqrt-units) and 44 on 1,226,096 (0.04 sqrt-units). The carrier is
   exactly conserved by the generator and not by any finite trajectory; the
   census says so instead of choosing a tolerance that hides it.
2. **Trap-channel crossing share vs reactive flux.** 0.11005 of forward
   crossings against 0.05837 of the reactive flux — off by 1.885×. Mechanism: a
   channel whose interior holds a metastable basin recrosses its own guard, so
   raw crossing counts over-represent it relative to committor-weighted flux.
   The two *direct* channels, which have no interior basin, agree to 0.81 %
   (the row above). This is a real property of the bed, not a defect: it is
   precisely why the itinerary cross-check in §9 is stated on the `lo:hi` ratio
   and not on the three-way share.

The test asserts that the census contains **at least one** non-conserving row —
a census with none has, per V-23, either checked nothing or omitted something —
and that every non-conserving row ships with its mechanism (`note` non-empty).

---

## 9. Guard itinerary — the strike observed, not derived

A third route to the strike, and the only one that runs the dynamics. 2000
walkers × 6000 steps, seed 0. The simulation reads `bed["q"]` only for side
labels and never reads a flux, so the agreement is a check and not an identity.

| T | forward crossings (lo / hi / trap) | empirical `lo:hi` | committor `lo:hi` | closed-form | gap |
|---|---|---|---|---|---|
| `0.9 T*` | 189,865 / 158,597 / 77,122 | 1.19715 | 1.19581 | 1.19581 | 0.11 % |
| `1.1 T*` | 232,957 / **268,759** / 111,354 | **0.86679** | 0.86389 | 0.86389 | 0.34 % |
| `0.25` | 26,666 / 2,462 / 3,602 | 10.83103 | 10.91963 | 10.91963 | 0.81 % |

At `1.1 T*` the empirical ratio is **below 1**: simulated walkers cross the
five-fold *higher* barrier more often than the single lower one. The strike is
observed in the trajectory, not only derived from the rate law. `direct = 0` at
every temperature — no crossing bypasses a guard.

---

## 10. Morse census

Lower-link criterion on the graph (a 1-complex, so the only indices are 0 and
1): empty lower link → minimum; lower link with `c ≥ 2` components → `c − 1`
index-1 critical points; `c = 1` → regular point.

| landscape | nodes | edges | `m₀` | `m₁` | regular | `χ = V − E` | `b₀` | `b₁` |
|---|---|---|---|---|---|---|---|---|
| default | 11 | 16 | **3** | **8** | 0 | **−5** | 1 | 6 |
| `V(C) → 2.5` (must-fire) | 11 | 16 | **2** | **7** | 2 | **−5** | 1 | 6 |

`m₀ − m₁ = χ` on both. Morse inequalities `m₀ ≥ b₀ = 1` and `m₁ ≥ b₁ = 6` hold
on both. The must-fire raises `C` above its flanking saddles, turning it from a
minimum into an index-1 point and demoting `S_ac`/`S_cb` to regular points: the
**classifier moves** (3/8 → 2/7) while the **invariant cannot** (χ stays −5). A
census that returned constants would fail the first half; one that miscounted
components would fail the second.

---

## 11. Limits

The committor of the default landscape is dyadic and temperature-independent, a
consequence of Metropolis's unit downhill rates on a symmetric graph. That makes
the closed-form check exact and the harmonic residual identically zero, but it
means the default bed does not exercise a general non-dyadic solve; the
`jitter = 0.05` landscape is what does, and it is the one whose residual
(1.04e-17) should be read as this module's numerical accuracy. The `q = ½` guard
partition is not generating on this bed (deficit 11.9 % of `h`), so R5's policy
reads distribution-predict rather than point-predict here; a bed with a finer
guard alphabet would need to be built to move that. The CK, Pesin and itinerary
batteries are single-seed at 1.2e7 samples — the thresholds carry 2×–128× of
margin over the measurements, but no seed-to-seed spread is reported, and R5's
`N = 8` seeds are not run here. The bed is a generator and its instruments only:
nothing trains, and no arm has been scored on it.
