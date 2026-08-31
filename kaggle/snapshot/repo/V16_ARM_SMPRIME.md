# V16 SATURN (it. 5) — ARM S-M′, THE HOP BUILT AS THE PATH PRODUCT

Node `SATURN`, iteration 5 of CEQ v16 (the Banking Round). Discharges
`CEQ_V16_CONTRACT.md` PART VI it.5 — the §S-M′ arm, its three identity binds,
their planted negatives, and the manifests. Answers the coordinator's three
routes to the same wall.

**Files written.** `ceq/arm_smprime.py` (new), `tests/arm_smprime/conftest.py`,
`tests/arm_smprime/test_arm_smprime.py` (new), this file. Nothing under
`ceq/arm_phase.py`, `ceq/arm_pl.py`, `scripts/`, `scale/`, `lean/`,
`MISTAKES.md`, or `tests/` outside `tests/arm_smprime/` was touched. No git
command that writes was run. Repo HEAD at run time
`bbbc68832b056570353b12de6e1e1dd41f2833c7`; `git status --porcelain` shows only
`ceq/arm_smprime.py`, `tests/arm_smprime/` and the harness's own
`house-events.jsonl`. torch `2.5.1+cu121`, `torch.get_num_threads() = 20`,
devices `cpu` and `cuda` (RTX 4060 Laptop), float64 / complex128 for every
bind. **Nothing was trained.**

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) Does the oracle-gate bind hold on BED-M's REAL support?** | **YES, for the first time.** `residual = 5.919777e-16` at `n=64, s=64`; `7.550528e-16` at `n=512, s=64` and at `n=256, s=128`. Gates and drives read from `make_equilibrium_batch`, support `{−1, 0, +1}` confirmed on the draw, label recomputed by the bed's own `equilibrium_oracle` at float64. Bar `1e-6`; the residual is **ten orders inside it**, and its **real part is exactly `0.000e+00`** — the whole residual is the `1.2246e-16` imaginary dust `polar(1, π)` injects. |
| **(b) Reachable parameters** | `(u, θ) = (1, π) → −1.0 + 1.2246467991473532e-16j`, `(0, ·) → 0.0 + 0.0j`, `(1, 0) → 1.0 + 0.0j`. **Real part exact at all three; modulus exact at all three (`1.0`, `0.0`, `1.0`).** |
| **(c) Three corners** | `β=1, g≡0, QK-on` **bitwise** against `#5a`'s own `softmaxAttn`; `β=0, g≡0, QK-on` **bitwise** against `exp(qk)`; `β=0, QK-off` **bitwise** (cpu) against an independent double-loop path product. Against `ceq/lm.py`'s `Attention("softmax_x").operator` the softmax corner is **NOT bitwise — `1.110223e-16`, 19/64 entries** — and the mechanism is named in §4.2. |
| **(d) Corners distinct** | `\|c₁−c₂\| = 4.472918`, `\|c₁−c₃\| = 1.144938`, `\|c₂−c₃\| = 5.335671`. |
| **(e) Switches** | `Σ_j\|W_ij\| = 1.000000` at `β=1` on **every** row; at `β=0` the same rows read `[1.312192, 0.724290, 2.563817, 2.264559, 10.293107, 2.721943, 3.096841, 1.337183]`; at `β=0.5`, in between. `g`: `max\|g=1 − g=0\| = 0.673101`. `QK`: `max\|qk=1 − qk=0\| = 3.522037`. All three are `nn.Parameter`s on the shipped module. |
| **(f) Planted negatives** | Five, all fire. `drop_phase 1.934830`, `drop_magnitude 0.466267`, `beta_one 1.335288`, `exp_scan nan`, wrong-`β` corner `> 0.5`. Honest cell `1.110223e-16`. The manifest hash moves under all five. |
| **(g) 40 gradient steps from the initialiser** | **COMPLETES, 8 of 8 configurations**: `{cpu, cuda} × {float32, float64} × {as-constructed, identity_heads()}`, no non-finite gradient in 40 steps. `V16_DEVICE_CERT.md` §5.4 measured `ceq/arm_phase.py` failing at **step 0** as-constructed on both devices. |
| **(h) What `m = 1` does** | Nothing, **at the corner the bind is claimed at**. `V = b` unrescaled, so `log(1−m)` and `1/(1−m)` are never evaluated; band draw (`\|a\| = 1` everywhere) reads `residual = 1.110223e-16`, `v_max = 2.094085` finite. At `β = 1` the label bind **fails at `1.335288`** and no finite compensation exists at `m = 1` — that half is unmoved, and it is a property of the `β=1` corner, not of the closed interval. §6. |
| **(i) Diagnostic** | Annihilation MCC. **Corpus alone `1.000000`** (`fp = fn = 0`), **zero-step arm ×8 `0.639571`**, **`exp_scan` must-fire `0.000000`** with `0` predicted positives. Headroom `0.360` between floor and ceiling and `0.640` between the must-fire and the floor. The gate-`R²` instrument is reported beside it and reads `1.000000` corpus-alone as `V15_ARM_PHASE.md` §8 says it must. §8. |
| **(j) Did anything need a new construction?** | **NO.** Every object is a proved statement of `lean/CEQ/V16Domain.lean` evaluated directly — `#2` re-stated clauses 1 and 4, and `#5a`'s three corners. §2 states the correspondence line by line so a reader can check it. |
| **(k) Regression** | `tests/arm_smprime` **35 passed** on cpu and **35 passed** on cuda, same ids. `tests/loop` **15 failed / 523 passed** with this node's files against **15 failed / 522 passed** without, measured back to back at this HEAD; **same fifteen by name**. `tests/arm_phase` **30 passed**, unchanged. |

---

## 0. RED

Tests were written first. At that point `ceq/arm_smprime.py` did not exist:

```
=================================== ERRORS ====================================
___________ ERROR collecting tests/arm_smprime/test_arm_smprime.py ____________
ImportError while importing test module '...\tests\arm_smprime\test_arm_smprime.py'.
Traceback:
tests\arm_smprime\test_arm_smprime.py:46: in <module>
    from ceq import arm_smprime as smp
E   ImportError: cannot import name 'arm_smprime' from 'ceq'
=========================== short test summary info ===========================
ERROR tests/arm_smprime/test_arm_smprime.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.64s
```

**Four tests then went RED against the first written implementation. Two were
implementation defects the tests caught; two were claims weakened to what is
true.** Recorded because which is which is the whole value of writing them
first.

| test | what it found | resolution |
|---|---|---|
| `..._planted_negatives_all_fire[exp_scan]` | `exp_scan did not fire: 1.1102230246251565e-16` — `label_cell` computed the route but **did not pass `route=` to `readout`**, so the mutilation was silently discarded and the negative read as the honest cell | **implementation fixed.** A planted negative that cannot reach the object it mutilates is `V-24`'s empty rejection region with the sign flipped |
| `..._the_label_holds_at_both_closed_endpoints` | `a.abs().min()` read `nan`, because the draw poisoned the BOS gate with `nan` (`ceq/arm_phase.py::draw`'s convention). The poison also made the `exp_scan` negative fire **from the BOS instead of from the corpus's zeros** | **draw changed**, BOS gate set to the identity `1`, and `test_the_bos_gate_is_never_read` added to check the BOS is unread by moving it to `−7.25+3.5j` and reading the residual unchanged. The negative now fires for the theorem's reason |
| `..._path_product_corner_is_bitwise` (cuda only) | `torch.cumprod` is sequential on cpu and a **parallel scan on cuda**, so the vectorized route re-associates: `11 / 64` entries move by at most `5.551115e-17` | **claim weakened to what is true**, pinned per device, and the zeros — the clause the label bind depends on — asserted equal on both |
| `..._the_association_order_is_named...` | on a generic complex draw the two multiplication orders differ at `6.206335e-17` (`83 / 256` entries) | **claim weakened**, and the order the definition uses is now stated in the module docstring |

A fifth defect was found by the 40-step probe during design rather than by a
test, and it is in §7.

## 0b. GREEN

```
$ python -m pytest tests/arm_smprime -q
................
  path-product corner  cpu    entries moved 0/64   max|gap| 0.000000e+00
.........
  40-step gradient probe  cpu   float32  as-constructed  -> first non-finite: None
  40-step gradient probe  cpu   float32  identity        -> first non-finite: None
  40-step gradient probe  cpu   float64  as-constructed  -> first non-finite: None
  40-step gradient probe  cpu   float64  identity        -> first non-finite: None
..
  annihilation MCC, CORPUS ALONE (no arm)   = 1.000000   (positives 257664/266240)
  annihilation MCC, ZERO-STEP ArmSMPrime x8 = 0.639571   per seed [0.706199, 0.814533,
      0.426074, 0.556524, 0.714698, 0.872869, 0.746152, 0.27952]
  annihilation MCC, exp_scan MUST-FIRE      = 0.000000   (predicted positives 0)
.
  gate R^2, CORPUS ALONE (no arm)           = 1.000000
  gate R^2, ZERO-STEP ArmSMPrime x8         = 0.291617   per seed [0.196315, 0.146722,
      0.246818, 0.594783, 0.086057, 0.322347, 0.095066, 0.64483]
....                                      [100%]
35 passed in 6.10s

$ ARM_SMPRIME_DEVICE=cuda python -m pytest tests/arm_smprime -q
................
  path-product corner  cuda   entries moved 11/64   max|gap| 5.551115e-17
.........
  40-step gradient probe  cuda  float32  as-constructed  -> first non-finite: None
  40-step gradient probe  cuda  float32  identity        -> first non-finite: None
  40-step gradient probe  cuda  float64  as-constructed  -> first non-finite: None
  40-step gradient probe  cuda  float64  identity        -> first non-finite: None
..                                                      [identical MCC and R^2 rows]
....                                      [100%]
35 passed in 5.16s
```

**Every number except the path-product association split is identical on the
two devices**, and that split is a summation-order property with the zeros
unmoved.

---

## 1. THE CONSTRUCTION, AS BUILT

```
    W_ij  =  G_ij · exp(qk · q_i·k_j) / Z_i^β                j ≤ i, 0 above
    G_ij  =  Π_{k=j+1}^{i} m_k e^{i θ_k}                     THE PATH PRODUCT
    R_ij  =  Π_{k=j+1}^{i} m_k                               its modulus, clause 1
    Z_i   =  Σ_{j≤i} R_ij exp(qk · q_i·k_j)                  the modulus row
    m     =  clamp(lerp(1, u, g), 0, 1)                      CLOSED, cap LAST
    θ_eff =  θ · g
```

**`β`, `qk` and `g` are the three switches**, and `#5a`'s corners are settings
of them: softmax at `β=1, g=0, QK-on`; linear attention at `β=0, g=0, QK-on`;
the path product at `β=0, QK-off`.

**`Z_i` is real and strictly positive by construction**, because the diagonal
term is the empty product `R_ii = 1` times `exp(w_ii) > 0`. So `Z^β` needs no
positivity side condition and `Z^0 = 1` for every `Z` — which is how both `β=0`
corners come out bitwise rather than nearly.

**The division is done on the real and imaginary parts separately.** Dividing a
complex tensor by a real one promotes and runs the general complex quotient
`(ac+bd)/(c²+d²)`; that is not bitwise `a/c` even at `d = 0`, and the two
bitwise corners would not be.

**The upper triangle is masked BEFORE the exponential**, with `−inf`, so
`exp(−inf) = 0` exactly and its backward is `grad·0`. Masking after would create
`inf` above the diagonal to meet a zero gradient, which is a `nan`.

**Parameters.** `ArmSMPrime` **4,806** against the softmax control's `4,769`
(`V15_R1.md`), ratio **`0.007758`**, inside the 10% bar. Two per-position heads
(`u`, `θ`) and three scalar switches, against ARM PHASE's three heads: **there
is no key-bias head**, because at `β = 0` the row is not a softmax that needs
compensating.

---

## 2. THE CORRESPONDENCE TO `V16Domain.lean`, LINE BY LINE

The round's first law strikes any new construction. **This module is the
computational shadow of statements already proved**, and the mapping is printed
here so a later reader can check the claim rather than take it.

| module object | Lean statement it evaluates |
|---|---|
| `gate(m, θ) = m·polar(1, θ)` | `gateOf (m θ : ℝ) : ℂ := (m : ℂ) * Complex.exp ((θ : ℂ) * Complex.I)` — verbatim |
| `path_product(gate(m, θ))` | `pathProd m θ i j := ∏ k in Ico (j+1) (i+1), gateOf (m k) (θ k)` — **the definition, with no logarithm in it** |
| `path_product(m)` (the real route) | clause 1's right-hand side, `Complex.abs (pathProd m θ i j) = ∏ k in Ico (j+1) (i+1), m k` |
| zeros of `path_product` | clause 4, `(∃ k ∈ Ico (j+1) (i+1), m k = 0) → pathProd m θ i j = 0` — a **theorem**, which is why the hop annihilates exactly |
| `hop_scan` | the object `no_prefix_scan_represents_a_zero_gate` refutes, kept as the shipped planted negative |
| `operator(β=1, g=0, QK-on)` | `corner_softmax` |
| `operator(β=0, g=0, QK-on)` | `corner_linear` |
| `operator(β=0, QK-off)` | `corner_path_product`, **repaired**: Lean's corner 3 is `CEQ.V15.Wc` — the real exponential form — and `path_product_corner_fails_at_a_zero_gate` proves it wrong at a zero gate for any `g`. §1's clause-4 object is what replaces it |
| `Σ_j \|W_ij\| = 1` at `β=1` | `softmax_row_sum_one` / `beta_one_row_is_one` |
| `Σ_j \|W_ij\| ≠ 1` at `β=0` | `gate_zero_beta_zero_row_not_one` |
| `magnitude = clamp(·, 0, 1)` | the `[0,1]` **CLOSED** hypothesis of `prefix_logit_mask_restated` (`h0 : ∀ k, 0 ≤ m k`, `h1 : ∀ k, m k ≤ 1`) |
| `(u,θ) = (\|a\|, arg a)` | `bedM_gate_exact`, `negative_draw_is_on_the_band` |

**The one thing this file does that Lean does not name** is the evaluation
strategy — a masked reverse cumulative product with the association order fixed
right-to-left. That is an implementation of `Finset.prod`, not a mechanism: it
adds no term, no parameter and no branch, and §4.4 measures what the choice of
order costs (`6.206335e-17` off BED-M's support, **`0.000000e+00` on it**).

---

## 3. BIND 1 — ORACLE GATES IN ⇒ LABEL ≤ 1e-6, ON BED-M's ACTUAL SUPPORT

**This is the bind that has never held.** `V15_ARM_PHASE.md` (e) reads `nan` on
the band; `MISTAKES.md` V-25 records why: the identity's hypothesis is
`|a| < 1`, the corpus supplies `|a| ∈ {0, 1}`, and `0 of 2,048` sequences
satisfy the quantified hypothesis at any seed or size.

### 3.1 On the real corpus

Gates and drives are `x[:, :, CH_DRIVE]` and `x[:, :, CH_FLIP]` from
`make_equilibrium_batch(t_star=2)`; the label is the bed's own
`equilibrium_oracle`, **recomputed at float64** so the residual is the arm's and
not the corpus's float32 storage. The setting is the path-product corner:
`β = 0`, QK off, `(u, θ) = (|a|, arg a)`, `V = b` **unrescaled**.

| shape | support read on the draw | zero-hop fraction | residual | `\|Re\|` part | `\|Im\|` part |
|---|---|---|---|---|---|
| `n=64, s=64` | `[−1.0, 0.0, 1.0]` | `0.967788` | **`5.919777e-16`** | `0.000e+00` | `5.920e-16` |
| `n=512, s=64` | `[−1.0, 0.0, 1.0]` | `0.967788` | **`7.550528e-16`** | `0.000e+00` | `7.551e-16` |
| `n=256, s=128` | `[−1.0, 0.0, 1.0]` | `0.984133` | **`7.550528e-16`** | `0.000e+00` | `7.551e-16` |

**The real part is exactly zero at every shape.** The entire residual is
imaginary, and it is `polar(1, π)`'s `1.2246e-16` accumulated along the path.
BED-M's label is real; a reader who takes `.real` gets it bitwise.

**Why this works where the exponential route cannot**, in one line: the label
is `Σ_j (Π_{k>j} a_k) b_j` — `equilibrium_oracle`'s own docstring calls it "the
signed path sum" — and at `β = 0` with QK off the operator **is** `Π_{k>j} a_k`.
There is no softmax row to compensate, so there is no `s_j = log(1−m_j)` and no
`V_j = b_j/(1−m_j)`, and the two expressions that are singular at the closed
endpoints are never written.

### 3.2 Reachability, exhibited

`V16_LEAN_DOMAIN.md` (e) proves `#6` covers **0 of 3** of BED-M's values, so an
arm with `#6`-parametrized gates cannot be *set* to BED-M's oracle gates at all.
The re-stated parametrization can be, and the parameters are:

| target | `(u, θ)` | what float64 returns | `\|a\|` |
|---|---|---|---|
| `−1` | `(1, π)` | `−1.0 + 1.2246467991473532e-16j` | `1.0` |
| `0` | `(0, ·)` | `0.0 + 0.0j` | `0.0` |
| `+1` | `(1, 0)` | `1.0 + 0.0j` | `1.0` |

Real part exact at all three, modulus exact at all three. `bedM_gate_exact`
proves this exact in ℂ; the `1.2246e-16` is `polar`'s, not the theorem's, and it
is the whole of §3.1's residual.

### 3.3 Both closed endpoints, on one draw

`bedm_draw` carries `m = 0` on the first third of the row (the builder's own
head zeroing) and `m = 1` with `θ ∈ {0, π}` after it, so a single cell exercises
both endpoints. Residual **`1.110223e-16`**. The band draw (`|a| = 1`
everywhere, BED-M's *propagate* support) reads the same, with `v_max = 2.094085`
finite.

### 3.4 Planted negatives — the rejection region

| mutilation | residual | manifest hash |
|---|---|---|
| `none` (the honest cell) | `1.110223e-16` | `3994d8f4c0a2…` |
| `drop_phase` (θ → 0, the sign carrier deleted) | **`1.934830`** | `31908ca8e880…` |
| `drop_magnitude` (m → 1, the annihilator deleted) | **`0.466267`** | `04951ac79de1…` |
| `beta_one` (the row normalized) | **`1.335288`** | `79262e607227…` |
| **`exp_scan`** (the forbidden route) | **`nan`** | `0dc0165e01a0…` |

**`exp_scan` lives in `ceq/arm_smprime.py`, not in the test file.** V-24's
requirement is that the rejection region be occupied by shipped code, and the
code that occupies it here is the construction the theorem forbids. The test
asserts `not (residual <= 1e-6)` **and** `isnan`, so a `nan` cannot be read as a
pass in either direction.

The manifest moves under all four: `cell_manifest` folds an `SMP_FIELDS` block
(`variant`, `route`, `beta`, `qk`, `g`, three head settings, `m_max`, `v_max`,
`n_zero_gates`) into `identity_manifest`'s base hash, and `device` is read off
the tensors — `record["device"] = a.device.type`, live, per
`V16_R1_DEVICE_READY.md`.

---

## 4. BIND 2 — THE THREE CORNERS OF `#5a`

### 4.1 Bitwise against `#5a`'s own statements

Probe: `q, k ~ N(0,1)` at `[8, 4]`, seed 3, float64.

| corner | setting | reference, written from the theorem | result |
|---|---|---|---|
| softmax | `β=1, g≡0, QK-on` | `exp(w_ij) / Σ_{j'≤i} exp(w_ij')` | **bitwise**, `imag ≡ 0` bitwise |
| linear attention | `β=0, g≡0, QK-on` | `exp(w_ij)` | **bitwise**, `imag ≡ 0` bitwise |
| the path product | `β=0, QK-off` | explicit double loop, stated order | **bitwise** (cpu); cuda in §4.3 |

### 4.2 Against `ceq/lm.py`, and the gap is named rather than rounded

The reference is `ceq.lm.Attention("softmax_x", 4, 1).operator` — the repo's
**inclusive-causal** control, `SMX_TAU = SMX_RHO = 1`. `ceq/bench.py` is not
used anywhere in this node: it masks with `tril(-1)`, so `A_ii = 0` and row 0 is
empty, and binding the corner claim to one object and the label claim to another
is V-24's shared-carrier defect.

```
  corner1 vs lm.Attention("softmax_x"): bitwise False   max|gap| 1.110223e-16   19/64 entries
```

**It is not bitwise, and that is a structural statement about §S-M′, not a
tolerance.** `torch.softmax` is a fused kernel that subtracts the row maximum;
`#5a`'s `softmaxAttn` does not. Measured on the same probe, **no**
expression-level route reproduces the fused kernel bit for bit:

| route | bitwise | `max\|gap\|` | entries moved |
|---|---|---|---|
| unshifted ratio (**what this arm computes**) | no | `1.110223e-16` | `19 / 64` |
| max-shifted ratio (torch's own algorithm, written out) | no | `1.110223e-16` | `22 / 64` |
| max-shifted, reciprocal-multiply | no | `1.110223e-16` | `23 / 64` |

So an operator that materializes `Z_i^β` cannot be bitwise against it. `ceq/arm_phase.py` is bitwise **because it calls
`torch.softmax`**, which it can only do by putting the gate in the logit, in the
log domain, which is the route `no_prefix_scan_represents_a_zero_gate` forbids
on BED-M.

**So the two properties cannot both be had by one evaluation route, and this is
the trade the round should have on record:** bitwise-against-torch's-softmax
costs the band; the band costs `1.110223e-16` — a quarter of one ulp of `1.0` —
against `lm.py`, while staying bitwise against the theorem `#5a` actually
states.

### 4.3 The device split, reported not widened

| device | entries moved vs the loop reference | `max\|gap\|` | zeros agree |
|---|---|---|---|
| cpu | `0 / 64` | `0.000000e+00` | yes |
| cuda | `11 / 64` | `5.551115e-17` | **yes** |

`torch.cumprod` is sequential on cpu and a parallel scan on cuda, so the
vectorized route re-associates. Same class as `V15_ARM_PHASE.md`'s
`n_exactly_one` split (`9767` cpu, `7713` cuda). **The zeros are identical on
both devices**, which is the clause the label bind depends on, and that is
asserted rather than the gap being widened.

### 4.4 The association order, stated and priced

A complex product is not associative in float64, so the definition fixes one
order: `path_product` accumulates `k = i` down to `k = j+1`.

| draw | ascending vs descending |
|---|---|
| generic complex, `\|a\| ∈ (0,1)`, `s = 16` | `max\|gap\| = 6.206335e-17`, `83 / 256` entries move |
| **BED-M's own support `{−1, 0, +1}`** | **`max\|gap\| = 0.000000e+00`, `0 / 289` entries move** |

On the corpus the bind is claimed at, the order is not a free parameter.

### 4.5 `corners_are_distinct`, in float64

`|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`, `|c₂−c₃| = 5.335671`. A containment
whose corners coincide is decoration; these do not coincide.

> **PLANTED NEGATIVE, BIND 2** — the corner with the wrong switch. `β = 0` where
> the softmax corner claims `β = 1`: `max|gap| > 0.5` against `lm.py`. Gate left
> **on** at `β = 1`: `max|gap| > 0.1`. So the corner is a statement about `β`
> **and** about `g`, not about either alone.

---

## 5. BIND 3 — `β`, `QK` AND `g` ARE PRESENT AND EFFECTIVE

### 5.1 `β` decides softmax-class membership

`#5b`: `β`, not `g`, is the switch that decides membership. Read as the row sum
of the modulus row, which is the probability vector:

| `β` | `Σ_j \|W_ij\|`, rows `0..7` |
|---|---|
| `0.0` | `1.312192, 0.724290, 2.563817, 2.264559, 10.293107, 2.721943, 3.096841, 1.337183` |
| `0.5` | `1.145510, 0.851052, 1.601192, 1.504845, 3.208287, 1.649831, 1.759784, 1.156366` |
| **`1.0`** | **`1.000000` on every row** (`max\|Σ−1\| < 1e-14`) |

The same separation holds at the `g ≡ 0` corner on the real row sum, so the
reading is about `β` and not about the gate.

### 5.2 `QK` and `g`

| switch | test | result |
|---|---|---|
| `QK` off | operator must not depend on `q, k` **at all** | `torch.equal` across two independent `(q,k)` draws — **True** |
| `QK` on vs off | | `max\|Δ\| = 3.522037` |
| `g` off | must be the `g ≡ 0` corner **exactly**: `m ≡ 1`, `θ ≡ 0`, hop = the all-ones causal mask | `torch.equal(mod, tril(ones))` and `torch.equal(hop.real, tril(ones))` — **True** |
| `g` on vs off | | `max\|Δ\| = 0.673101` |

All three are `nn.Parameter`s on `ArmSMPrime` with `requires_grad`, and setting
each to `0` in turn moves `forward(x)` — so they are switches the harness can
train, not constructor arguments.

> **PLANTED NEGATIVE, BIND 3** — the exponent frozen. If `Z_i^β` were written
> `Z_i` (the switch present in the signature, absent from the arithmetic), the
> mutilated operator agrees with the `β=1` reading to `< 1e-14` and differs from
> the `β=0` reading by `> 0.1`. So the mutilation is exactly *"β has no effect"*,
> and the bind's `β=0` row is what rejects it.

---

## 6. THE THIRD ROUTE TO THE WALL — WHAT THE TWO CLOSED ENDPOINTS COST HERE

The coordinator names three independent findings against `[0,1]` CLOSED. This
node's reading is that **the closure is not the defect; two logarithms are, and
they sit at opposite ends.**

| end | the singular expression | where it lives | this arm |
|---|---|---|---|
| `m = 0` | `log m = −inf`, backward `1/0` | `ceq/arm_phase.py:121,128`, the **gate scan** | **absent.** `path_product` is a cumulative product; `#2` re-stated defines it with no logarithm and clause 4 makes `m=0` a theorem instead of a hole |
| `m = 1` | `log(1−m) = −inf`, `1/(1−m) = inf` | `ceq/arm_phase.py::oracle_heads`, the **value rescale** | **absent at the corner the bind is claimed at.** At `β = 0` the row is the path product, so `V = b` unrescaled and neither expression is evaluated |

Measured, so the claim is not rhetorical:

```
  log(1-m) at m=1 : -inf        1/(1-m) at m=1 : inf
  band draw |a| = 1 everywhere, this arm:  residual = 1.110223e-16   v_max = 2.094085
```

### 6.1 The 40-step probe

`V16_DEVICE_CERT.md` §5.4's probe, same 40-step budget, random data, first step
at which any gradient over the arm's parameters goes non-finite.

`n = 512, s = 64`, `lr = 1e-3`, AdamW, run through the shipped
`arm_smprime.gradient_finiteness`.

| arm | init | cpu fp32 | cpu fp64 | cuda fp32 | cuda fp64 | positions at `m = 0` after |
|---|---|---|---|---|---|---|
| `arm_phase` (`V16_DEVICE_CERT.md` §5.4) | as constructed | **step 0** | — | **step 0** | — | `50.0763%` cpu / `49.8627%` cuda |
| `arm_phase` | `identity_heads()` | **step 14** | — | **step 23** | — | `0.0031%` |
| **`arm_smprime`** | **as constructed** | **none in 40** | **none in 40** | **none in 40** | **none in 40** | **`55.5817%`, all four** |
| **`arm_smprime`** | **`identity_heads()`** | **none in 40** | **none in 40** | **none in 40** | **none in 40** | `0.0000%` |

**8 of 8 configurations complete, and the last column is why the comparison is
fair rather than lucky: this arm sits with `55.58%` of its positions on the
CLOSED lower endpoint — MORE than the `50.08%` that kills `arm_phase` at step 0
— and takes 40 steps anyway.** The endpoint is not the failure; `log` of it is. The mechanism is exactly the one the
coordinator names: `arm_phase` fails because roughly half a randomly-initialised
magnitude head clamps to the closed lower endpoint and `log(0)` is `−inf`; this
arm takes no logarithm of the magnitude, so that step does not exist.

Nothing is fitted by the probe and nothing is kept — no cell, no checkpoint, no
loss curve. An initialiser that produces `−inf` is an identity failure, which is
why it is in this node's bind table and not in a training report.

### 6.2 The ruling this node does **not** make

`V15_X36_PRIOR_ART.md` prescribes `[0,1)`. **This node does not change the
contract's `[0,1]`, and it reports that under the path-product route the
prescription is not needed**: both of the singularities `[0,1)` exists to avoid
are properties of `arm_phase`'s **evaluation route** (a scan of `log m`, and a
softmax row compensated by `1/(1−m)`), not of the interval. Under §S-M′-as-
path-product, `[0,1]` closed reproduces BED-M's label at `5.9e-16` and completes
40 gradient steps on both devices at both dtypes.

**The half that is unmoved:** at `β = 1` the label bind fails at `1.335288`, and
no finite `V` recovers it at `m = 1`, because the softmax row has divided out
the magnitudes the label needs. So the `m = 1` problem is not solved in general
— it is **confined to the `β = 1` corner and measured there**, and the corner
the identity bind is claimed at does not instantiate it. Whether R1′ trains an
arm that drifts to `β = 1` is a question for it.7 and this node does not answer
it.

**The one-character fix and its evidence, for the author's ruling.** If the
round nonetheless wants `[0,1)`, the change is `torch.clamp(x, 0.0, 1.0)` →
`torch.clamp(x, 0.0, 1.0 - eps)` in `magnitude`. The evidence **against** making
it: `bedM_gate_exact` needs `m = 1` to represent `a = ±1`, which is `2 of 3` of
BED-M's support and `2 of 2` of `make_propagate_batch`'s — a half-open cap
cannot be set to BED-M's oracle gates, which is the same defect
`V16_LEAN_DOMAIN.md` (e) proves against `#6`, one endpoint over. The evidence
**for**: `V15_X36_PRIOR_ART.md`'s `73,766 / 200,000` clipped draws landing on
exactly `1.0`, which is only a defect where `1/(1−m)` is evaluated.

---

## 7. THE DEFECT THE 40-STEP PROBE FOUND IN THIS ARM, AND ITS FIX

Recorded because it is this node's own mistake and the mechanism generalizes.

The `g` switch blends a raw head toward the identity gate. The first
implementation clamped **first** and blended second:

```python
m = torch.lerp(torch.ones_like(u), torch.clamp(u, 0., 1.), g)      # WRONG
```

At `g > 1` that **extrapolates past the lower endpoint**: with `clamp(u) = 0`
and `g = 1.01`, `m = 1 + 1.01·(0−1) = −0.01`. A negative magnitude makes the
modulus row negative, and `Z^β` for a negative base and non-integer `β` is
`nan`. Measured: `Zmin = −0.001891`, first non-finite gradient at **step 32 of
40**, fp32, cpu, `AddmmBackward0`.

```python
m = torch.clamp(torch.lerp(torch.ones_like(u), u, g), 0., 1.)      # the fix
```

**The cap goes last**, so `m ∈ [0,1]` is a property of the arithmetic at every
value of the switch — which is what a CLOSED cap is for, and the first version
had it as a property of `g` staying in range instead.
`test_the_magnitude_never_leaves_the_closed_interval_through_the_switch` sweeps
`g ∈ {−2, −0.5, 0, 0.5, 1, 2, 25}` over 4,096 draws of `u ∈ [−3, 3]`.

Two earlier iterations of the same probe are worth recording as measurements
even though they are gone from the shipped module:

| version | first non-finite | mechanism |
|---|---|---|
| `Z = num.abs().sum(-1)` over the **complex** numerator | **step 11 of 40** (fp32) | `AbsBackward0` returned `nan` — an inf gradient meeting `sgn(0) = 0` at the `96.9%` of entries that are exactly zero. Replaced by summing `path_product(m)`, which is **clause 1's own right-hand side**, so the normalizer never takes `abs` of a complex zero |
| clamp before blend | **step 32 of 40** (fp32) | §7 above |
| shipped | **none in 40**, all 8 configurations | |

**Both fixes replaced an operation with a proved identity rather than with a
guard.** Neither adds a term, a parameter or an epsilon.

**The gradient the closed cap still costs, unchanged from X₃₆.**

```
  u      -5.0   -1e-9    0.0    0.5    1.0   1+1e-9    2.0   285.0719
  m       0.0     0.0    0.0    0.5    1.0      1.0    1.0        1.0
  dm/du   0.0     0.0    1.0    1.0    1.0      0.0    0.0        0.0
```

`V15_ARM_PHASE.md` §7 item 7's dead zone is inherited verbatim and is not
repaired here. **What is NOT inherited** is gradient death at the annihilating
gate itself: `d(Σ G)/dm` at `m = [1, 0, 0, 0.7, 0.9]` reads
`[0.0, 1.0, 2.33, 1.9, 1.7]` — the two zero positions receive `1.0` and `2.33`,
because `torch.cumprod`'s backward handles a single zero in a window exactly.
`log m`'s backward at `m = 0` is `1/0`.

---

## 8. THE DIAGNOSTIC, RUN ON THE CORPUS ALONE AND AT ZERO STEPS

L-DIAG permits a contract to say **what** a diagnostic must distinguish, never
**which statistic** does it. What this one must distinguish is the thing this
node's whole finding is about: **an arm whose hop can be exactly zero from one
whose hop cannot.** The statistic was chosen on data, and all three controls
were read before it was trusted.

**Instrument: annihilation MCC.** Matthews correlation between `{the arm's hop
is exactly 0}` and `{the corpus's hop is exactly 0}`, over the `133,120` causal
pairs `j ≤ i` of BED-M `e3_t2` at `n = 128, s = 64`. MCC and not accuracy
because `96.78%` of causal pairs annihilate, so accuracy is `0.968` for the
constant predictor.

| reading | value | counts |
|---|---|---|
| **(a) corpus alone, no arm** (the bed's own `CH_DRIVE` drives the hop) | **`1.000000`** | `fp = 0`, `fn = 0`, positives `257,664 / 266,240` |
| **(b) zero-step `ArmSMPrime`, 8 seeds** | **`0.639571`** | per seed `0.706199, 0.814533, 0.426074, 0.556524, 0.714698, 0.872869, 0.746152, 0.279520` |
| **(c) must-fire: the `exp_scan` route** | **`0.000000`** | **`0` predicted positives** — `Complex.exp` is never zero |

**Three readings.**

1. **The ceiling is `1.000000` and the mechanism is the same one M-21 names.**
   The gate is an input channel (`scale/negation_scope.py:432`), so an
   instrument fed the corpus's own gate recovers the annihilation set exactly.
   Any reading short of `1.0` measures the arm's parametrization, never the
   availability of the information. This is stated up front rather than
   presented as a strength.

2. **It has headroom, and it has a hard must-fire the gate-`R²` does not.**
   Floor `0.639571`, ceiling `1.000000` — `0.360` of room, less than the gate
   `R²`'s `≈0.70`. What it has instead is `(c)`: an arm that takes the
   prefix-scan route reads **exactly `0.000000`** however good its gates are,
   because its predicted positive set is empty. The gate `R²` is blind to that
   — it scores the gate, not whether the hop can carry it — so the two
   instruments are complements and both are reported.

3. **The gate `R²` is carried forward beside it, and its zero-step reading is
   identical to ARM PHASE's to the last printed digit.**

| reading | this node | `V15_ARM_PHASE.md` §8 |
|---|---|---|
| gate `R²`, corpus alone, no arm | `1.000000` | `1.000000` |
| gate `R²`, zero-step ×8 seeds, `n = 512` | **`0.291617`** | **`0.291617`** (ARM PHASE) |

The match is not a coincidence and it is worth one line: `ArmSMPrime` creates
`wq, wk, m_head, theta_head` in the same order `ArmPhase` does, so under
`torch.manual_seed(seed)` the two arms draw the **same bytes** into the two
heads the feature is computed from. A future comparison of trained gate `R²`
across the two arms therefore shares a baseline rather than needing two.

**Nothing in this section is trained.** Only the two controls and the must-fire
exist at this node.

---

## 9. DEVICE — AND A SECOND PAYOFF FROM THE SAME REPAIR

`V16_R1_DEVICE_READY.md` records that `cumsum_cuda_kernel` has no deterministic
implementation in torch 2.5.1, so `use_deterministic_algorithms(True)` — which
`scale/r10_capacity_sweep.py::main()` sets unconditionally on its cuda path —
makes both existing arms **raise** on cuda. Re-measured here, with the shipped
functions rather than with bare kernels:

```
  arm_smprime.operator        under use_deterministic_algorithms(True) on cuda: OK
  arm_smprime.path_product    under use_deterministic_algorithms(True) on cuda: OK
  arm_phase.operator          under use_deterministic_algorithms(True) on cuda: RAISES
      RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
  arm_phase.scan_phase        under use_deterministic_algorithms(True) on cuda: RAISES
```

**`cumprod` has a deterministic CUDA kernel; `cumsum` does not.** The path
product replaced the scan for a theorem's reason, and the determinism conflict
`V16_DEVICE_CERT.md` §5.3 prices goes away as a side effect. This is reported
as a measurement, not as an argument for a device decision — NEPTUNE owns that.

**No memory figure is reported by this node.** `V16_DEVICE_CERT.md` §3 measures
the sizing module at `m/p = 1.842` for the complex arm — under-predicting, the
direction a sizing gate must never have — with `C_OPERATOR` at `7.50` for
`8 B/element`. Any cell priced on this arm must measure rather than model, and
this file quotes no prediction from it.

---

## 10. REGRESSION

```
$ python -m pytest tests/arm_smprime -q                       35 passed in 6.10s
$ ARM_SMPRIME_DEVICE=cuda python -m pytest tests/arm_smprime -q
                                                              35 passed in 5.16s
$ python -m pytest tests/arm_phase -q                         30 passed in 1.66s
```

`tests/loop`, measured **A/B in the same session at this HEAD**, this node's two
paths absent and then present:

```
--- WITHOUT this node's files ---   15 failed, 522 passed, 3 warnings in 34.18s
--- WITH    this node's files ---   15 failed, 523 passed, 3 warnings in 30.10s
```

**The failure set is fifteen and it is the same fifteen by name** — ten
`test_no_test_file_imports_conftest_as_a_bare_module` parametrizations over
`tests/chase/` and `attic/tests/chase/`, two
`test_corpus_is_recoverable_and_verifiable`, one
`test_no_corpus_instance_has_a_boundary_node_that_cannot_propagate[reproduce-n1024-d4-t0.95]`,
one `test_no_weight_record_omits_a_declared_identity_field`, one
`test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on`. The `+1`
passing case is
`tests/loop/test_no_module_writes_a_file_at_import.py::test_module_has_no_import_time_write_or_argv_read[ceq/arm_smprime.py]`,
confirmed by collection.

**The circulating baseline is stale again.** `V15_ARM_PHASE.md` §9 recorded
`15 failed / 517 passed` without its files and flagged the brief's `515` as
stale; at this HEAD the same measurement reads **`15 failed / 522 passed`**. The
failure count and the fifteen names are what the check is about and both are
unchanged; the passed count has moved `515 → 517 → 522` across three HEADs.
**Quote the failure set, not the pass count.**

---

## LIMITS

- **Nothing here is trained.** Every number is a float64/complex128 identity
  residual, a bitwise comparison, a zero-step control, or a finiteness verdict.
  R1′'s pre-registered `8/8 converge` and the author's counter are **not**
  tested by this node and are not commented on.
- **The `m = 1` problem is confined, not solved.** §6 shows the identity bind's
  corner does not instantiate `1/(1−m)`; it does **not** show that a trained arm
  stays at that corner. At `β = 1` the label bind fails at `1.335288` and no
  finite compensation exists at `m = 1`. If R1′ pins `β = 1`, that half of the
  defect returns and this file does not price it.
- **The softmax corner is not bitwise against `ceq/lm.py`** (§4.2), by
  `1.110223e-16` over `19/64` entries. It is bitwise against `#5a`'s own
  statement of softmax. Any table quoting "bitwise softmax" for this arm must
  say which of the two objects it means.
- **The path-product corner's association differs by device** (§4.3): `11/64`
  entries, `5.551115e-17`, on cuda. The zeros agree on both, and the label bind
  depends only on the zeros — but a future bitwise claim across devices for the
  full complex operator would be false.
- **The oracle-gate bind is an EXPRESSIBILITY statement.** It says the arm can
  be *set* to reproduce BED-M's label from BED-M's own gates. It says nothing
  about whether an optimizer finds those gates, and the `θ ∈ {0, π}` values it
  needs sit at two isolated points of a free real parameter.
- **The diagnostic's ceiling is `1.000000` for M-21's reason** — the gate is an
  input channel. Its floor `0.639571` leaves `0.360` of headroom, **less than
  the gate `R²`'s**, and it is an observation-grade instrument; this node does
  not promote it to a kill condition. Its value over the gate `R²` is the
  must-fire, not the headroom.
- **The 40-step probe is a finiteness verdict on random data, not evidence about
  optimization.** It shares `V16_DEVICE_CERT.md` §5.4's budget, optimizer and
  shape (`n = 512, s = 64`, AdamW, `lr = 1e-3`) so the two tables are
  comparable; the in-suite copy runs the cheaper `n = 128, s = 32` and both were
  run on both devices.
- **`torch.cumprod`'s backward at a window with two or more zeros returns `0`
  for the positions inside it.** §7 measures the single-zero case receiving real
  gradient; the multi-zero case does not, and the census on BED-M `e3_t2`
  (`n=64, s=64`) reads `0` zeros on `4,288 / 133,120` causal windows
  (`3.2212%`), exactly `1` zero on `4,032` (`3.0288%`) and **`≥ 2` zeros on
  `124,800` (`93.75%`)**. That is a mechanism note against a future training
  run, not a second prediction (M-20).
- **No memory or wall-clock figure is quoted** (§9), because
  `V16_DEVICE_CERT.md` §3 measures the sizing module under-predicting this class
  of arm by `1.842×`.
- **`ceq/lm.py`, `ceq/arm_phase.py` and `scale/` were read, never written.**
  `chain_label` and `band_draw` are imported from `ceq/arm_phase.py` unchanged,
  so the label and the band draw are the same objects the phase arm published
  against.

---

**Distance to the north star.** The §S-M′ hop now reproduces BED-M's own label
from BED-M's own gates at `5.919777e-16` against a `1e-6` bar, with the real
part exactly zero — a bind that read `nan` in every previous filing, on a corpus
where `96.8%` of hops annihilate and `0 of 2,048` sequences satisfy the
hypothesis the previous theorem was proved under. The three corners of `#5a` are
bitwise against the theorem's own statements and pairwise distinct at `O(1)`;
`β`, `QK` and `g` are live parameters and `β` separates the row sums `1.000000`
from `10.293107`. The arm completes 40 gradient steps from its own initialiser
in all eight device/dtype/init configurations where the phase arm fails at step
0, and it is legal under the determinism flag on cuda where the phase arm
raises. **The claim sentence these binds permit is: the hop can now represent
the corpus's annihilating gate exactly, and the value path still cannot be
normalized without giving that back.**
