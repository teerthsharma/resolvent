# V15 X₃₇ — the three topological certificates

`CEQ_V15_3_DELTA.md` §X₃₇. Node: SATURN-5, CEQ v15.3 round.

Files added: `ceq/certs/__init__.py`, `ceq/certs/topological.py`,
`tests/certs/test_topological.py`. Nothing else in the tree was touched.
`ceq/rips.py`, `ceq/beds/**`, `scale/**`, `lean/**` and `MISTAKES.md` are
byte-identical to their pre-round state; `ceq/arm_phase.py` belongs to another
node and was not read into scope. `tests/beds`: **28 passed**, unchanged.

`tests/certs`: **36 passed in 6.56s**. Nothing trains. float64 throughout, and
§9 states where that is and is not enough.

---

## 0. What `ceq/rips.py` gives, and the one thing it does not

The delta says the `β₁` certificate goes *"via the author's Rips toolkit"*, so
the first job was to read what the toolkit is. `ceq/rips.py` is a geodesic
Vietoris–Rips **generator on `S²`**: a `SplitMix64` sampler, the edge rule
`dot ≥ cos r`, a critical-bridge edge, and a component labeller. That is a Rips
**1-skeleton** and **`β₀`**. It builds no 2-simplices, so on its own it computes
**no `β₁`, no persistence diagram and no bottleneck distance.**

Stating that is not a complaint about the toolkit — it is a port of a MuJoCo
benchmark generator and was never a homology engine. It decides the honest
division of labour, which is:

| what | where from |
|---|---|
| the **metric** | `ceq.rips`'s own geodesic rule, `d = acos(x·y)` on `S²` |
| the **filtration** | `ceq.rips.rips_edges`, arithmetic unchanged; `toolkit_edges` only inverts its degree parameter, `deg = (n−1) sin²(r/2)`, into the radius `r = 2 asin(√p)` the rule already computes internally |
| `β₀` at any radius | `ceq.rips.components` |
| `β₁` at any radius | flag complex on the toolkit's edge set, `β₁ = (E − V + β₀) − rank ∂₂` — 20 lines added on top, `toolkit_flag_betti` |
| the **barcode** and the **bottleneck distance** | `ripser==0.6.14` / `persim==0.3.8`, already pinned in `requirements.txt` and already calibrated against six invariants in `tests/foreman/test_topology_washout.py` |

The two are not assumed to agree. `test_the_toolkit_edge_rule_and_the_geodesic_metric_are_the_same_filtration`
asserts, at six radii, that the toolkit's `dot ≥ cos r` edge set is exactly
`{(i,j) : d_geo(i,j) ≤ r}` — so the barcode engine and the toolkit are shown to
filter the same complex — and `test_the_authors_toolkit_alone_reproduces_the_barcode_verdict`
re-reads `β₁` at five radii through the toolkit path alone (§4.3).

---

## 1. RED

`tests/certs/test_topological.py` was authored and run before
`ceq/certs/topological.py` existed.

```
$ python -m pytest tests/certs -q
=================================== ERRORS ====================================
______________ ERROR collecting tests/certs/test_topological.py _______________
ImportError while importing test module 'C:\Users\seal\Desktop\New folder (32)\tests\certs\test_topological.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\certs\test_topological.py:27: in <module>
    from ceq.certs import topological as T
E   ImportError: cannot import name 'topological' from 'ceq.certs' (unknown location)
=========================== short test summary info ===========================
ERROR tests/certs/test_topological.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.47s
```

## 2. GREEN

```
$ python -m pytest tests/certs -v
collected 36 items

test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[-3]           PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[-1]           PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[0]            PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[1]            PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[2]            PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[5]            PASSED
test_winding_is_an_exact_integer_on_a_known_k_turn_sequence[15]           PASSED
test_winding_refuses_an_undersampled_sequence_and_the_unguarded_read_is_wrong  PASSED
test_winding_refuses_a_path_that_does_not_close                          PASSED
test_the_step_guard_alone_does_not_catch_every_alias_and_refinement_does  PASSED
test_winding_refuses_inputs_it_cannot_read[bad0..bad3]                    PASSED
test_winding_refuses_a_refinement_that_is_not_finer                      PASSED
test_order_parameter_locks_and_matches_the_wrapped_normal_closed_form     PASSED
test_order_parameter_must_fire_on_phases_that_carry_no_order              PASSED
test_the_toolkit_edge_rule_and_the_geodesic_metric_are_the_same_filtration PASSED
test_persistent_beta1_is_one_on_rps_and_zero_on_the_coordination_control  PASSED
test_the_persistence_threshold_is_geometric_and_never_reads_the_barcode   PASSED
test_the_authors_toolkit_alone_reproduces_the_barcode_verdict             PASSED
test_the_barcode_moves_by_at_most_two_epsilon[0.001, 0.01, 0.05]          PASSED
test_the_toolkit_alone_shows_the_two_epsilon_interleaving_on_beta0[0.001, 0.01] PASSED
test_euler_poincare_equals_the_bed1_poincare_hopf_index_sum              PASSED
test_b0_is_computed_and_not_assumed_to_be_one                            PASSED
test_the_bed1_morse_function_is_nondegenerate_and_the_complex_is_closed   PASSED
test_the_census_moves_and_the_invariant_cannot                           PASSED
test_poincare_hopf_transversality_fails_at_mu_zero[0.0, 0.01, 0.1]        PASSED
test_the_index_sum_is_pinned_to_chi_only_once_the_field_points_inward     PASSED
test_the_rps_orbit_is_not_chaotic_and_its_horizon_is_stated               PASSED
test_winding_and_beta1_agree_per_instance                                PASSED

============================= 36 passed in 6.56s ==============================
```

### 2b. Mutation-checked, because 36/36 on the first implementation run is not evidence

`V15_BED1.md` §3b's discipline, applied to this suite. Eight mutants introduced
one at a time into `ceq/certs/topological.py`, suite re-run under `-x`, module
restored from a backup each time.

| mutant | first test killed |
|---|---|
| M1 `winding` drops the `max_step` guard | `..._refuses_an_undersampled_sequence_...` |
| M2 `winding` rounds instead of refusing a non-integer | `..._refuses_a_path_that_does_not_close` |
| M3 `order_parameter` averages moduli (`≡ 1`) instead of the resultant | `..._locks_and_matches_the_wrapped_normal_closed_form` |
| M4 `toolkit_edges` ignores the radius | `..._toolkit_edge_rule_and_the_geodesic_metric_are_the_same_filtration` |
| M5 flag `β₁` skips `rank ∂₂` | `..._authors_toolkit_alone_reproduces_the_barcode_verdict` |
| M6 `β₀` assumed to be 1 | `test_b0_is_computed_and_not_assumed_to_be_one` |
| M7 `replicator_field` drops the mutation term | `..._transversality_fails_at_mu_zero[0.01]` |
| M8 `replicator_index_sum` skips the transversality refusal | `..._index_sum_is_pinned_to_chi_only_once_the_field_points_inward` |

No mutant survived. **M6 is the one that matters**: it is BED-1's own
`morse_census` behaviour (`betti_0 = 1` returned as a constant), and it is
killed only by the test that ranges over a *disconnected* graph. On BED-1's
landscape nothing can tell the constant from a computation, which is exactly why
§6.2 flags it.

---

## 3. Certificate (a) — `Z` winding, per instance

`winding_number(theta, closed=True, max_step=π/2, refinement=None) -> int`.
Returns Python `int`, never a float. Reads the sequence as a **cycle**, which is
the only reading under which winding is an integer.

### 3.1 Correct on a known `k`, including negative

64 samples across `k` turns. The raw `Σ wrapped Δ / 2π` is printed beside the
returned integer, because the delta calls a non-integer reading an instrument
defect and the claim is checkable only if the pre-rounding value is shown.

| `k` | returned | raw `Σ Δ / 2π` | `max\|Δ\| / π` |
|---|---|---|---|
| −3 | **−3** | `-3.0` | 0.093750 |
| −1 | **−1** | `-1.0` | 0.031250 |
| 0 | **0** | `0.0` | 0.000000 |
| +1 | **+1** | `1.0` | 0.031250 |
| +2 | **+2** | `2.0000000000000004` | 0.062500 |
| +5 | **+5** | `5.0` | 0.156250 |
| **+15** | **+15** | `15.0` | **0.468750** |

`k = 15` is the closest passing case to the `π/2` refusal, and `k = 16` lands on
it exactly. The largest departure from integrality anywhere in the sweep is
`4.4e-16` at `k = 2`, against an `INTEGRALITY_TOL` of `1e-9` — seven orders of
margin.

### 3.2 The refusal on aliased input, and the unguarded reading beside it

**3 turns in 5 samples.** Every true step is `1.2 π`, which wraps to `−0.8 π`.

```
max|wrapped step| = 0.8000 pi
REFUSED: undersampled: largest wrapped step 0.8000 pi exceeds 0.5000 pi, so the
         phase aliases and the winding is not determined by these samples
unguarded reading (max_step=pi, the naive Nyquist limit) = -2      true k = 3
```

`-2` is a plausible integer and it is wrong. The unguarded path is kept
reachable via `max_step=math.pi` rather than deleted, so the defect stays
measurable rather than merely described — `MISTAKES.md` M-1's repair pattern.

**A path that is not a cycle.** Read open, a half turn gives exactly `0.5`:

```
REFUSED: total turn 0.5 is not an integer within 1e-09; an open path has no
         integer winding
```

Read closed, the *same* input is refused for the other reason — closing a half
turn is a step of `π`, squarely in the aliasing band. Both rejection regions
fire at `O(1)`.

### 3.3 The step guard is necessary and **not sufficient** — a defect found in this module's own first design

The `π/2` guard was written, tested green, and is wrong as a complete guard. A
wrapped step of magnitude `s` is consistent with true steps
`s, s ± 2π, s ± 4π, …`; nothing computed from one sampling can separate them.
Measured here:

| true `k`, 64 samples | `max\|Δ\| / π` | guarded reading |
|---|---|---|
| 33 | 0.968750 | REFUSED |
| **63** | **0.031250** | **−1** |
| 65 | 0.031250 | +1 |
| 127 | 0.031250 | −1 |

**63 turns sampled 64 times has the healthiest-looking largest step in this
whole file — `0.0313 π` — and the certificate returned `−1`.** That is a V-16
instrument reporting a pass, produced by this node's own guard.

The repair is a refinement check, and it is the only thing that can close the
gap: pass a finer sampling of the same path and require the two readings to
agree.

```
n =   64 samples -> -1
n = 1024 samples -> 63
REFUSED: aliased: 64 samples read -1 and 1024 samples of the same path read 63;
         the reading is not stable under refinement
```

It is an **empirical** sufficiency check, not a proof, and the docstring says
so: no reading from finitely many samples can be more than that. Every winding
in §6.1's per-instance table is refinement-certified against the raw integrator
output, which is 20×–25× finer than the reported sampling.

---

## 4. Certificate (b) — persistent `β₁` of a carrier trajectory

**Recurrent iff `β₁ ≥ 1`.** The carriers are replicator flows on the 2-simplex,
radially projected onto `S²` (a homeomorphism, the inverse gnomonic map) so the
toolkit's geodesic rule applies, then resampled to 48 points **uniform in arc
length** — a reparameterisation, which changes no topology and makes the
sampling scale a resolution rather than an artefact of where the field runs
slowly.

### 4.1 The threshold, frozen before the run (MISTAKES.md M-2)

A `β₁` bar that dies immediately is noise, and a threshold picked after seeing
the bars is not a threshold. This one is

```
tau(X) = STABILITY_FACTOR * delta_max(X),    STABILITY_FACTOR = 4.0
delta_max(X) = max_i min_{j != i} d_geo(i, j)
```

**The 4 is derived, not tuned.** An unmatched bar `(b, d)` sits `(d−b)/2` from
the diagonal in the bottleneck metric, and Rips persistence obeys
`d_B ≤ 2 d_H`, so a Hausdorff perturbation of size `h` can create or destroy any
bar with `d − b ≤ 4h`. `delta_max` bounds the Hausdorff distance between the
sample and the curve it was drawn from. Both factors come from the theorem and
from the point **positions**; neither reads the barcode.
`test_the_persistence_threshold_is_geometric_and_never_reads_the_barcode` pins
that by rotating and permuting the cloud and requiring `tau` not to move.

### 4.2 The must-fire, `[RUN: 1 vs 0]`

| carrier | `delta_max` | `tau` | `H₁` bars (birth, death) | persistence | `β₁` | recurrent |
|---|---|---|---|---|---|---|
| **RPS**, cyclic dominance | 0.045392 | 0.181569 | `(0.045414, 0.551497)` | **0.506083** | **1** | **True** |
| **coordination**, the delta's named control | 0.019912 | 0.079649 | *none* | 0.000000 | **0** | **False** |

**1 vs 0, both halves.** The RPS bar clears its threshold by **2.787×**. The
coordination control produces **no finite `H₁` bar at any radius**, so the
verdict does not depend on where the threshold sits.

The two flows: `A_RPS` is the zero-sum cyclic-dominance payoff, whose orbits are
closed because its rows *and* columns sum to zero; `A = I` is coordination,
whose barycentre is a repeller and whose interior orbits run monotonically to a
vertex — a path, not a cycle.

### 4.3 The same verdict through the author's toolkit alone

`ceq.rips.rips_edges` + `ceq.rips.components` + `rank ∂₂` over the 3-cliques of
the toolkit's own edge set. No `ripser` on this path.

| radius | | `β₀` | `β₁` | `E` | triangles | `rank ∂₂` |
|---|---|---|---|---|---|---|
| 0.022707 | `0.5 × birth` | 48 | **0** | 0 | 0 | 0 |
| 0.068121 | `1.5 × birth` | 1 | **1** | 48 | 0 | 0 |
| 0.298455 | mid-bar | 1 | **1** | 303 | 810 | 255 |
| 0.523922 | `0.95 × death` | 1 | **1** | 621 | 3726 | 573 |
| 0.606646 | `1.1 × death` | 1 | **0** | 768 | 5952 | 721 |

`0 → 1 → 1 → 1 → 0` across the bar the engine reported. Two instruments, no
shared code below the edge rule, same answer.

### 4.4 Stability

`ceq/rips.py` **exposes no bottleneck distance** — it has an edge rule and a
component labeller and nothing else — so the number comes from `persim`.
Perturbation is a per-point rotation on `S²` through an angle drawn from
`U(0, ε)`, which makes `d_H ≤ ε` hold **exactly in the metric the filtration
uses**; perturbing in `R³` and renormalising would not give that in closed form.

| `ε` | measured `d_H` | bottleneck | bound `2ε` | ratio |
|---|---|---|---|---|
| 1e-3 | 9.715607e-04 | **1.261592e-03** | 2.000000e-03 | 0.631 |
| 1e-2 | 9.715607e-03 | **1.067853e-02** | 2.000000e-02 | 0.534 |
| 5e-2 | 4.857804e-02 | **5.480166e-02** | 1.000000e-01 | 0.548 |

**The sharpest form the toolkit can state unaided.** It has `β₀` and nothing
else, and `β₀(·, r)` is non-increasing in `r`, so a `2ε`-interleaving reads as a
sandwich:

```
b0(X, r + 2 eps)  <=  b0(X_eps, r)  <=  b0(X, r - 2 eps)
```

| `ε` | violations / radii | strict span | floor `4ε` | grid step |
|---|---|---|---|---|
| 1e-3 | **0 / 120** | 0.004806 | 0.004000 | 0.000534 |
| 1e-2 | **0 / 120** | 0.039518 | 0.040000 | 0.000534 |

`strict_span` is the measure of radii where the two ends of the sandwich
actually differ — without it a sandwich that never bites would report a pass
while measuring nothing (V-16). Its floor is the window width `4ε`, which is
what the readings sit at.

### 4.5 A third control the certificate does NOT reject, printed rather than omitted

Truncating the RPS orbit to a **300° arc** leaves an open curve with a 60° gap.

| carrier | `delta_max` | `tau` | bar | persistence | `β₁` | birth / `delta_max` |
|---|---|---|---|---|---|---|
| RPS, closed | 0.045392 | 0.181569 | `(0.045414, 0.551497)` | 0.506083 | 1 | **1.0005** |
| RPS, 300° arc | 0.039428 | 0.157713 | `(0.294044, 0.562953)` | 0.268909 | **1** | **7.457** |

**The certificate reads `β₁ = 1` on the arc, at 1.705× its threshold.** That is
correct persistent homology and a real limit on what the certificate claims: at
scales above the gap the arc *is* a loop, and `β₁ ≥ 1` certifies a cycle in the
trajectory's shape at the measured scale, **not that the trajectory is
periodic**. The quantity that separates them is birth relative to the sampling
scale — `1.0005` against `7.457`, the closed orbit being born at its own
resolution. That is **not** promoted into the certificate, because a criterion
chosen after seeing these two numbers would be M-2. It is stated as a limit and
left for a pre-registration to adopt.

---

## 5. Certificate (c) — Euler–Poincaré vs the Poincaré–Hopf index sum

### 5.1 Which object the cross-check can live on

The RPS carrier **cannot host it**, and the reason is structural rather than a
missing implementation:

* the Rips complex of a *trajectory* models the trajectory's image — a circle,
  `Σ(−1)^k β_k = 1 − 1 = 0`;
* the equilibrium census of a *vector field* lives on the field's domain — the
  simplex `Δ²`, a disk, `χ = 1`.

Comparing them would be a category error, not a measurement. Worse, the two
requirements are in direct opposition here: **the carrier that has `β₁ = 1` is
exactly the carrier where Poincaré–Hopf does not apply** (§5.3 — the pure
replicator's field is tangent to every face), and the field where it does apply
(`μ > 0`) spirals into the barycentre and has no persistent loop. The
cross-check therefore runs on **BED-1**, which is the object in this repo that
carries both an equilibrium census and a complex over the same space.

### 5.2 The comparison

`ceq.beds.bed_1.morse_census` classifies every node by the number of components
of its lower link and yields `m₀ − m₁`. This module counts components of the
whole graph through **`ceq.rips.components`** and forms `β₀ − β₁`. Neither reads
the other.

| landscape | `V` | `E` | `m₀` | `m₁` | regular | index sum `m₀ − m₁` | `β₀` | `β₁` | `Σ(−1)^k β_k` | `χ` | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 11 | 16 | **3** | **8** | 0 | **−5** | 1 | 6 | **−5** | −5 | **AGREE** |
| must-fire `V(C) → 2.5` | 11 | 16 | **2** | **7** | 2 | **−5** | 1 | 6 | **−5** | −5 | **AGREE** |

**Agreement, not a mismatch.** `Σ(−1)^k β_k = −5` matches `V15_BED1.md`'s
`χ = −5 = m₀ − m₁` on both landscapes, and the classifier moving `(3,8) → (2,7)`
while the invariant cannot is reproduced from the Betti side.

**No census defect was found on BED-1.** Two weaknesses in the surrounding
apparatus were, and they are §6.

### 5.3 The hypotheses, checked rather than assumed

`MISTAKES.md:1406` records Poincaré–Hopf invoked where the field is tangent at
`μ = 0`. Both hypotheses are tested on data here.

**Transversality.** A graph is a closed 1-complex: `boundary_cells = 0`, so the
transversality hypothesis is **vacuous** on BED-1 — there is no boundary for a
field to be tangent to. That is *why* the cross-check is well posed there and
not on the simplex.

**Non-degeneracy**, which is not vacuous. An edge whose endpoints carry the same
value has no lower link and the census is undefined on it. Measured:
`degenerate_edges = []` on both landscapes. The five `S_hi` saddles share
`V = 2.0` but are pairwise non-adjacent, so no edge is degenerate.

**The prior strike, reproduced on its own object.** On the face `z₀ = 0` of the
simplex, at `x = (0, 0.6, 0.4)`:

| `μ` | `dz₀` | |
|---|---|---|
| 0.00 | **`+0.000000e+00`** | **TANGENT — the face is invariant, the hypothesis fails** |
| 0.01 | `+3.333333e-03` | inward |
| 0.10 | `+3.333333e-02` | inward |

Identical in magnitude to `attic/workdonenew.pre-v13.md:201`'s
`−0.000000e+00 / +3.333333e-03 / +3.333333e-02`. (The sign of the zero differs
only because this implementation always adds the `μ(1/n − x)` term, and
`−0.0 + 0.0 = +0.0` in IEEE754; the value is exactly zero either way.)

`replicator_index_sum` therefore **refuses** at `μ = 0`:

```
REFUSED: the field is tangent to the boundary (inward component 0.000000e+00 at
         mu=0.0); Poincare-Hopf does not apply and no index sum is stated
         -- MISTAKES.md:1406
```

and reports only once the field points inward:

| `μ` | inward margin | interior equilibria | indices | `Σ index` | `χ(Δ²)` |
|---|---|---|---|---|---|
| 0.02 | 6.666667e-03 | 1 | `[+1]` | **1** | **1** |
| 0.10 | 3.333333e-02 | 1 | `[+1]` | **1** | **1** |

Equilibria are located by Newton from a 24×24 grid on the simplex plane,
deduplicated, and restricted to the interior; the index is `sign det J` on the
2-D reduction, with a refusal if `|det J| < 1e-12` (a degenerate equilibrium has
no index).

---

## 6. Two weaknesses found, neither of them a defect in BED-1's answer

### 6.1 On a 1-complex, Euler–Poincaré cannot fail — only Poincaré–Hopf can

For a graph, `β₁` is the cycle rank `E − V + β₀`, so

```
b0 - b1 = b0 - (E - V + b0) = V - E = chi
```

**identically.** Euler–Poincaré on a chain complex is rank–nullity, a theorem,
not a measurement — the `Σ(−1)^k β_k = χ` half of certificate (c) is true for
any graph whatsoever and has no rejection region. The falsifiable content of (c)
is entirely in the other equality, `m₀ − m₁ = χ`, which is Poincaré–Hopf for a
discrete Morse function and which **does** have a rejection region: BED-1's own
must-fire landscape moves `(m₀, m₁)` and the suite requires the relation to
survive it. `euler_poincare_graph`'s docstring says this at the point of use
rather than leaving the reader to infer that half the certificate is a
tautology.

### 6.2 `bed_1.morse_census` returns `betti_0` and `betti_1` as constants

`ceq/beds/bed_1.py:374` reads `betti_0=1, betti_1=n_edges - n + 1`. Neither is
computed. On BED-1's landscape both are **correct** — recomputing `β₀` here
through `ceq.rips.components` returns `1`, so §5.2's agreement stands and no
number in `V15_BED1.md` §10 moves. But:

* the bed's own `χ` check therefore cannot detect a disconnected landscape;
* and `test_the_census_moves_and_the_invariant_cannot` would pass against
  hardcoded Betti numbers, which would make the cross-check a restatement rather
  than a check.

Recomputing `β₀` from the edge list is what makes §5.2 a cross-check. The
mutation table's **M6** plants exactly BED-1's behaviour (`β₀ := 1`) and it is
killed only by `test_b0_is_computed_and_not_assumed_to_be_one`, which ranges
over a *disconnected* graph — a triangle plus a disjoint edge plus an isolated
node, where `β₀ = 3`, `β₁ = 1`, `χ = 2` and the assumed form would report
`β₁ = −1`. **Nothing on BED-1's own landscape can tell the constant from a
computation**, which is the whole reason the test had to range off it.

No change is proposed to `ceq/beds/bed_1.py` — it is outside this node's write
scope, and its answer on its own bed is right.

---

## 7. Kuramoto / order parameter — `r = |1/N Σ e^{iθ_j}|`, both halves

### 7.1 `r → 1` under phase locking, against a closed form

Checked against the wrapped-normal resultant `exp(−σ²/2)` rather than against
"`r` is large", so the reading ranges over a different set than its own output
(`MISTAKES.md` V-3).

| phases | measured `r` | closed form | `\|diff\|` |
|---|---|---|---|
| wrapped normal `σ = 0.02`, `N = 65536` | 0.999800 | `exp(−σ²/2) = 0.999800` | 2.24e-07 |
| wrapped normal `σ = 0.05` | 0.998743 | 0.998751 | 7.85e-06 |
| wrapped normal `σ = 0.20` | 0.980110 | 0.980199 | 8.87e-05 |
| identical phases, `N = 1024` | **1.000000000000000** | 1 | 0 |

### 7.2 MUST-FIRE — `r → 0` on phases carrying no order

An order parameter that reads high on random phases is measuring nothing.

| control | `r` |
|---|---|
| `N = 4096` **equispaced** phases (deterministic, exact) | **5.551115e-17** |
| `N = 4096` **uniform**, 8 seeds | **0.007794 – 0.022456**, mean 0.012916 |
| Rayleigh prediction `E[r] = √π / (2√N)` | 0.013847 |
| refusal bar `5/√N` | 0.078125 |

The mean over 8 seeds lands within **6.7 %** of the Rayleigh prediction, and the
bar is fixed from `N` alone before the readings, not from the spread.

The equispaced control is the sharper of the two: it is deterministic, its true
value is exactly 0, and it reads `5.6e-17`. An instrument that returned
`mean(|e^{iθ}|)` instead of `|mean(e^{iθ})|` reads `1.000000` on it — mutant M3,
killed.

---

## 8. Per-instance table — (a) and (b) side by side

Printed per instance, as the delta asks. `winding` and `β₁` are computed by
disjoint code paths and must agree on which carriers cycle; every winding is
refinement-certified against the raw integrator output (`n_fine`).

| instance | winding | `n` / `n_fine` | `β₁` | recurrent | persistence | `tau` | `r` | invariant drift | closure gap |
|---|---|---|---|---|---|---|---|---|---|
| RPS `x₀=(.5,.3,.2)` | **+1** | 48 / 1141 | **1** | True | 0.506083 | 0.181569 | 0.000145 | 2.820e-12 | 1.923e-04 |
| RPS `x₀=(.6,.25,.15)` | **+1** | 48 / 1214 | **1** | True | 0.681365 | 0.258022 | 0.000699 | 7.544e-12 | 1.378e-03 |
| RPS reversed (`−A`) | **−1** | 48 / 1141 | **1** | True | 0.506094 | 0.181567 | 0.000145 | 2.850e-12 | 1.922e-04 |
| coordination | **0** | 48 / 983 | **0** | False | 0.000000 | 0.079649 | 0.995360 | n/a | n/a |

Notes on the table, because two rows would mislead if left bare:

* **`x₀=(.2,.5,.3)` would not be a second instance.** It has the same
  `x₁x₂x₃ = 0.03` as the first and therefore lies on the *same orbit*, one phase
  offset away — it returned a bit-identical persistence of `0.506083`. The
  second instance was moved to `(.6,.25,.15)` (`x₁x₂x₃ = 0.0225`), a genuinely
  different level set, which reports a different orbit size: `delta_max` 0.064505,
  persistence 0.681365.
* **`r` and winding measure different things, and the table shows it.** The
  cycling carriers have `r ≈ 1.5e-4` — their phases cover the circle uniformly,
  which is what a winding of `±1` *means* — while the coordination control is
  phase-locked at `r = 0.995360` and winds `0`. A high `r` is not evidence of
  recurrence, and this table is the demonstration.
* `invariant_drift` is reported `n/a` for coordination: `x₁x₂x₃` is conserved
  only when the payoff's columns sum to zero, and under coordination it runs to
  `0` at the vertex. Reporting a "drift" there would report the dynamics as an
  error.

---

## 9. The arithmetic statement (MISTAKES.md M-19)

M-19's horizon is `mantissa_bits / log2(stretching rate)` — about 52 float64
steps for a tent map. **It does not bind here, and this is a claim rather than
an omission.**

**Certificate (a) and the order parameter iterate nothing.** `winding_number`
reads a phase sequence it is given; `order_parameter` is a single sum. There is
no orbit and no horizon.

**Certificate (b)'s carrier is not chaotic.** `RPS_PAYOFF` has rows *and*
columns summing to zero, so `Σᵢ (Ax)ᵢ = 0` and `x·Ax = 0`: `x₁x₂x₃` and `Σx` are
**exact invariants of the flow**, the interior rest point is a centre, and every
Lyapunov exponent is 0. The stretching rate is 1, `log2(1) = 0`, and M-19's
horizon is unbounded. What *is* finite is RK4 truncation drift, so it is
measured rather than assumed:

| quantity | RPS, one turn | coordination |
|---|---|---|
| raw RK4 samples, `dt = 0.01` | 1141 | 983 |
| `x₁x₂x₃` drift, relative | **2.820198e-12** | n/a (not an invariant) |
| `\|Σx − 1\|` max | **8.881784e-16** | **8.881784e-16** |
| closure gap `\|x(T) − x(0)\|` | **1.922975e-04** | n/a (open path) |
| Lyapunov bound | **0.0** | contracting |

**The stated horizon is one turn**, 1141 steps, and the run stops there. The
closure gap `1.9e-04` is the direct measure of integration error over that
horizon and is the number a longer run would have to be scored against; it is
`4.24e-03` of the orbit's own `delta_max` — 236x below the sampling resolution
the barcode is read at, so it cannot move a bar. The
coordination carrier is contracting, not expanding, and stops when 99.9 % of its
arc length is behind it.

**Certificate (c) iterates nothing at all.** BED-1's own arithmetic statement is
`V15_BED1.md` §7 — an integer state sequence, closed-form entropies, no
real-valued map iterated. `morse_census` and `euler_poincare_graph` are integer
combinatorics on an 11-node graph. The replicator index sum of §5.3 is a Newton
solve on a 2-D reduction, not an orbit.

**Where float64 is not enough: nowhere in this file.** The two places it would
become the binding constraint are named rather than left implicit — a carrier
driven by a *chaotic* field (any positive Lyapunov exponent reinstates M-19 in
full and would need `scripts/v15_n1_probes/p08_pesin_symbolic.py`'s exact-rational
treatment), and a `rank ∂₂` at a radius where the flag complex is large enough
for the SVD's numerical rank to become ambiguous. The largest `∂₂` computed here
is `768 × 5952` with `rank = 721`, on a `±1` integer matrix, far from that.

---

## 10. Limits

The `β₁` certificate is read on **single trajectories at 48 arc-length points**;
no seed-to-seed spread is reported, and the two RPS instances differ by initial
condition rather than by noise. The 300° arc of §4.5 shows the certificate
reading `β₁ = 1` on a curve that is not periodic, so `recurrent` as defined here
means "carries a cycle at the measured scale", and the birth-to-resolution ratio
that would separate the two cases (`1.0005` against `7.457`) is reported but
deliberately not promoted into the rule. The persistence barcode and the
bottleneck distance come from `ripser`/`persim`, not from `ceq/rips.py`, which
exposes neither; the toolkit path is exercised at five radii and agrees, but it
is a spot check against a continuous barcode and not a second full computation.
`toolkit_flag_betti` computes Betti numbers over `Q`; these clouds are sampled
curves and carry no torsion, but the module does not check that. Certificate
(c) **agrees** on BED-1, so this node found no census defect there — half of
that agreement is a rank–nullity identity (§6.1) and the informative half is
`m₀ − m₁ = χ`, which is checked on two landscapes and no more. The
Poincaré–Hopf index sum of §5.3 locates equilibria by Newton from a 24×24 grid;
an equilibrium outside every basin of attraction of that grid would be missed,
and the certificate would then report a smaller index sum rather than refusing.
`winding_number`'s refinement check is empirical sufficiency, not proof. Nothing
in this file trains, and no arm has been scored on any of it.
