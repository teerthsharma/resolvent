# V15 MERCURY-4 — ARM PHASE, the amendment aimed at a measured failure

Node `MERCURY-4` of the CEQ v15.3 round. Builds `ceq/arm_phase.py` and
`tests/arm_phase/test_arm_phase.py` against `CEQ_V15_3_DELTA.md` §X₃₆, and
certifies five identity binds with a planted negative on each.

**The failure being fixed, precisely.** `V15_R1.md` §7 ran ARM PL at BED-M
`t* = 2, n = 2048, N = 8`: five seeds crossed `floor₁` with `â_max ∈ [1.10,
1.51]`, three diverged with `â_max` of `20.31`, `49.66`, `285.07` — above `1`,
where the `1/(1−a)` value rescale is undefined — and there is an order of
magnitude between the two populations with nothing in it. §X₃₆'s answer is
`a = m·e^{iθ}` with `m = clamp(u, 0, 1)`, a **hard cap**, so that `|a| ≤ 1` is
arithmetic and not luck.

**Nothing here trains** (L-LEAN). No optimizer, no gradient step, no cell. The
R1 re-run is a separate node and is gated on these binds.

**Files.** `ceq/arm_phase.py` (new, 4,820-param drop-in), `tests/arm_phase/`
(new: `conftest.py`, `test_arm_phase.py`), this file. Nothing under
`ceq/arm_pl.py`, `ceq/beds/`, `ceq/x35*/`, `scale/`, `lean/`, `tests/` outside
`tests/arm_phase/`, or `MISTAKES.md` was touched. No git command that writes was
run. Repo HEAD at run time `bce1556595799654a653761f02536eabdc202790`, torch
`2.5.1+cu121`, `torch.get_num_threads() = 20`, device `cpu`, float64 /
complex128 throughout.

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) Does `\|a\| ≤ 1` hold by construction?** | **YES.** Worst observed `\|a\|` is **`1.0` exactly** — the repr, not a rounding — over **2,200,000 draws**: 200,000 with `\|u\|` log-uniform on `[3.777513e-11, 2.647801e+10]` at both signs and `θ ∈ [−1000, 1000]`, plus 2,000,000 adversarial draws pinned at `m = 1`. `count(\|a\| > 1) = 0` in both. Also `1.0` at `u = ±inf`, `±1e308`, and at **R1's own eight `â_max` values**, `285.0719` included. |
| **(b) Band modulus digits** | `\|Π e^{iθ}\|` over `1e4` phases = **`1.000000000000`**, and the float64 repr is **`1.0`**. Independent cumulative-product route reads `0.9999999999999843`. Over all `1e4` prefixes: **`9767` read exactly `1.0`, `233` read one ulp low, `0` read above `1`.** |
| **(c) Parity / winding agreement** | **Exact, `torch.equal`.** `θ ∈ {0, π}` at `s = 64`: `0` of `4096` entries disagree with `lean/CEQ/V15.lean::parity_sign`'s mask. Winding integrality residual `2.131628e-14`. The continuous route `exp(i(Φᵢ−Φⱼ))` agrees to `6.762526e-14` and is **not** exact — that gap is reported rather than hidden behind the integer route. |
| **(d) Standard-attention identity** | Setting is **`m = 1, θ = 0, s = 0`**, i.e. `a = 1`, the cap's upper endpoint. `torch.equal(op.real, lm.Attention("softmax_x").operator)` = **True**; `op.imag` exactly zero; `op.real @ v` bitwise. **The complex-gemm read-out is NOT bitwise: `1.1102230246251565e-16`**, and that number is named rather than rounded to "bitwise". |
| **(e) Label bind** | **Holds on the open interior, fails on the band.** `9.155133597044475e-16` at `s = 8` and `5.2510145522368515e-15` at `s = 64`, against the contract's `1e-6` bar — for the **complex** chain, where ARM PL published `6.6613381477509392e-16` for the real one. On BED-M's band (`\|a\| = 1`) the residual is **`nan`**: `s_j = log(1−1) = −inf`, `V_j = b_j/0 = ±inf`. **That is the cost and it is not adjusted away.** |
| **(f) Planted negatives** | **Nine, all fire at `O(1)` or larger.** Cap dropped → `285.0719`. Band magnitude `0.9` → modulus `0.0`. Quarter turn → winding residual `0.500000`, `49.66%` of mask entries wrong. Identity setting moved by each of three heads → `0.562760 / 0.339569 / 0.478112`. Label mutilations → `0.838865 / 0.889267 / 1.000000 / 0.423865 / 0.933220`. |
| **(g) Gate `R²` instrument** | **Corpus alone, no arm: `R² = 1.000000`.** Zero-step `ArmPhase`, 8 seeds: **`0.300689`**. Zero-step `ArmPL` reproduced in the same process: `0.371583` against R1's published `0.371839`. The instrument has real headroom — floor `≈ 0.30–0.37`, R1's trained reading `0.699299`, ceiling `1.000000` — which is what the `sign(a)` probe did not have. |
| **Regression** | `tests/arm_phase` **30 passed**. `tests/loop` **15 failed / 518 passed** with this node's files against **15 failed / 517 passed** without them, measured back to back at this HEAD. The `+1` is `test_module_has_no_import_time_write_or_argv_read[ceq/arm_phase.py]`, passing. Same fifteen failures by name. `tests/arm_pl` **17 passed**, unchanged. |

---

## 0. RED

Tests were written first. At that point `ceq/arm_phase.py` did not exist:

```
=================================== ERRORS ====================================
_____________ ERROR collecting tests/arm_phase/test_arm_phase.py ______________
ImportError while importing test module 'tests\arm_phase\test_arm_phase.py'.
Traceback:
tests\arm_phase\test_arm_phase.py:45: in <module>
    from ceq import arm_phase, lm
E   ImportError: cannot import name 'arm_phase' from 'ceq'
=========================== short test summary info ===========================
ERROR tests/arm_phase/test_arm_phase.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.79s
```

**Two tests then went RED against a written implementation, and both were
findings rather than typos.** They are recorded because the fix in each case was
to weaken the claim to what is true, not to change the arm:

1. `torch.equal(readout(...).real, ref @ v)` → **False**. A complex gemm is a
   different reduction than a real one — `Re(Σ aᵢⱼvⱼ)` accumulates through
   `Re·Re − Im·Im` even when every `Im` is `0.0`. The operator is bitwise; the
   contraction against `V` is one ulp off. Test split in two and the ulp named.
2. `label_cell(band_draw(...))["residual"] > 1e-6` → **False, because the
   residual is `nan`**, not because it passed. `inf` values against zero
   softmax weights. The assertion is now `not (residual <= 1e-6)` plus an
   explicit `isnan`, so a `nan` cannot be read as a pass in either direction.

## 0b. GREEN

```
$ python -m pytest tests/arm_phase -q
....................
  m = 1 - eps     residual      v_max        1/(1-m)
  eps=0.01      1.332268e-15  2.094085e+02  1.000000e+02
  eps=0.0001    4.440892e-15  2.094085e+04  1.000000e+04
  eps=1e-06     2.220446e-15  2.094085e+06  1.000000e+06
  eps=1e-09     2.442491e-15  2.094085e+09  1.000000e+09
  eps=1e-12     6.883383e-15  2.094131e+12  1.000022e+12
.........
  gate R^2, CORPUS ALONE (no arm)  = 1.000000
  gate R^2, ZERO-STEP ArmPhase x8  = 0.291617   per seed [0.196315, 0.146722,
      0.246818, 0.594783, 0.086057, 0.322347, 0.095066, 0.64483]
.                                           [100%]
30 passed in 1.79s
```

---

## 1. THE CONSTRUCTION, AS BUILT

```
    a_j   =  m_j · exp(i θ_j),   m_j = clamp(u_j, 0, 1)          CLOSED
    C_j   =  Σ_{k≤j} log m_k  +  i Σ_{k≤j} θ_k                   prefix-PHASE
    l_ij  =  q_ij − Re C_j + s_j                       j = 0..i  (real)
    A_ij  =  softmax_j(l_i·) · exp(i (Im C_i − Im C_j))          (complex)
    O_i   =  Σ_{j≤i} A_ij V_j
```

**The query-side scan comes back, on the phase side only.**
`V15_JUPITER2_FORK.md` §8.4 item 3 deleted `C_i` from the logit because a causal
softmax annihilates any term constant across `j` in row `i`. That argument is
about the **logit**. `exp(i·Im C_i)` is a multiplicative factor **outside** the
normalizer, so it survives — and it has to, because the parity of the path
`j → i` is a function of both endpoints. The magnitude half is still key-only.
`VARIANT` records the pair as `key_only_mag+sym_phase` so a cell measured under
a key-only phase cannot be filed under this one.

**The row of moduli is a probability vector; the complex row is not.**
`Σ_j |A_ij| = 1` at every setting (`normalizer`, max `|Z − 1| =
4.440892098500626e-16`), which is what keeps the object inside the softmax
class. `Σ_j A_ij ≠ 1`, because the twist is the parity carrier and not a
probability. Stated here so no conservation column can quote the wrong one of
the two.

**Param count, for the drop-in claim.** `ArmPhase` **4,820**, `ArmPL` `4,803`,
softmax `4,769` (`V15_R1.md`). Ratio against the control **`0.010694`**, inside
the 10% bar. The `+17` over ARM PL is one `nn.Linear(16, 1)` — a third scalar
head, priced in §7.

---

## 2. BIND 1 — `|a| ≤ 1` BY CONSTRUCTION

This is the bind the whole delta rests on, so it is a **property test over a
range**, not three points. Three points is what V-24 calls an answer key.

| sweep | N | parameter range | worst `\|a\|` | `count(\|a\| > 1)` |
|---|---|---|---|---|
| general | 200,000 | `\|u\|` log-uniform `[3.777513e-11, 2.647801e+10]`, both signs; `θ ∈ [−1000, 1000]` | **`1.0`** | **0** |
| adversarial, pinned at the cap | 2,000,000 | `u = 1` exactly; `θ ∈ [−1000, 1000]` | **`1.0`** | **0** |
| float endpoints | 7 | `u ∈ {−inf, −1e308, −1, 0, 1, 1e308, +inf}` | `1.0` | 0 |
| **R1's own gate magnitudes** | 8 | `u =` the `â_max` column of `V15_R1.md` §7 | **`1.0`** | **0** |

The second sweep exists because `|a|` is computed by `hypot(m cos θ, m sin θ)`
and `cos² + sin²` is not exactly `1` in float64 — a cap that holds on the
parameter but leaks by one ulp on the modulus would still be a leak. It does
not: 2,000,000 draws at the worst case return exactly `1.0`.

**CLOSED, and the word is load-bearing.** Over `u ∈ [−2, 2]` at 401 points,
`clamp` returns `min = 0.0` and `max = 1.0` — both **attained**. The delta's
named prior art, **LRU (Orvieto 2023), `λ = exp(−exp ν + iθ)`**, evaluated on
the same grid returns `min = 6.179790e-04`, `max = 0.8734230184931167`; it
reaches neither endpoint, and `1 − max = 1.266e-01` at `ν = −2`. That is the
whole delta against LRU, measured on one grid rather than argued. **It matters
here specifically because BED-M's coefficients are `a ∈ {−1, 0, +1}`, i.e.
`m ∈ {0, 1}` — both endpoints — so an open magnitude cannot represent the
corpus it would be trained on.**

> **PLANTED NEGATIVE 1** — `gate(u, θ, cap=False)`, which is ARM PL's open
> magnitude, fed R1's own eight values: worst `|a| = **285.0719**`. The
> mutilation lives in the shipped module, not in a test-local copy, so the
> rejection region is occupied by the arm's own code.

---

## 3. BIND 2 — `|path product| = 1` EXACTLY ON THE BAND

`CEQ_V15_3_DELTA.md` records `[RUN: 1e4 phases → modulus 1.000000000000]`.
Reproduced, both routes, at `seed = 15`:

| reading | value |
|---|---|
| prefix-PHASE route, `\|exp(C_n − C_0)\|` | **`1.000000000000`** (repr `1.0`) |
| independent `numpy` cumulative complex product | `1.000000000000` (repr `0.9999999999999843`) |
| max over **all `1e4` prefixes** of `\| \|·\| − 1 \|` | `1.110223e-16` (one ulp) |
| prefixes reading **exactly** `1.0` | **`9767` / `10000`** |
| prefixes reading **above** `1.0` | **`0`** |
| max gap between the two routes | `1.565414e-14` |

**The `≤ 1` half survives float64 as well as the reals**: every one of the 233
misses is a miss *downward*. The endpoint modulus is exactly `1.0` at
`n = 10, 100, 1000, 10000, 100000` — deviation `0.000e+00` at every one. The
two routes share no code: one is `exp` of a `cumsum` in torch, the other a
`cumprod` in numpy, which is a check between two algorithms rather than one
identity restated twice.

> **PLANTED NEGATIVE 2** — `band_magnitude = 0.9`, the smallest possible
> departure from the cap's upper endpoint, over the same `1e4` positions:
> modulus **`0.0`**, deviation **`1.0`**. This is what makes the bind a
> statement about `m` and not about `exp(iθ)` alone.

---

## 4. BIND 3 — PARITY MASK = `Z₂` WINDING

The convention is the repo's own: `lean/CEQ/V15.lean::parity_sign`,
`χ(pscan p i − pscan p j) = ∏_{k=j+1}^{i} χ(p k)` with `χ` the sign character of
`ZMod 2`. `parity_sign_mask` implements it from the **theorem**, not from the
phase construction — deriving the reference from the object under test would be
V-3.

Setting `θ_k = π·p_k` and reading the winding `W_ij = (Φ_i − Φ_j)/π`:

| reading | value |
|---|---|
| winding integrality residual, `s = 64` | `2.131628e-14` |
| `torch.equal(parity_from_winding, parity_sign_mask)` | **True** |
| disagreeing entries | **`0` / `4096`** |
| per-instance winding `== cumsum(p)`, `s = 128` | **True**, residual `2.132e-14` |
| continuous route `max \|exp(i(Φᵢ−Φⱼ)) − χ\|` | `6.762526e-14` |

**The exactness is in the integer route only, and the difference is reported
rather than smoothed.** `cumsum` of `k` copies of `π` is not `fl(k·π)`, so
`exp(i(Φᵢ−Φⱼ))` misses `±1` by up to `6.76e-14`; rounding the winding to an
integer first recovers `±1` exactly. `CEQ_V15_3_DELTA.md` X₃₇ (a) says a
non-integer winding is an instrument defect, and `2.13e-14` is the number that
says which of the two happened here.

> **PLANTED NEGATIVE 3** — `θ ∈ {0, π/2}`, a `Z₄` gate instead of a `Z₂` one:
> winding residual **`0.500000`** (maximally non-integer) and **`49.66%`** of
> the `4096` mask entries wrong. The bind is a statement about the specific
> angles, not about "the phase carries a sign somehow".

---

## 5. BIND 4 — THE STANDARD-ATTENTION IDENTITY

**The setting, stated:** `m = 1` (any `u ≥ 1`; the module uses the magnitude
head's **bias** at `1.0`), `θ = 0`, `s = 0`. Equivalently `a = 1` — the cap's
**upper endpoint**, which is the same boundary ARM PL's `g ≡ 0` sat on
(`V15_JUPITER2_FORK.md` §8.4 item 5), now reached from inside a closed interval
instead of from an open one. At this setting `log m = 0.0` exactly, so the key
bias is exactly `0.0` and `w + 0.0` is bitwise `w`; and `Φ = 0`, so the twist is
exactly `1 + 0j`.

**The reference is `ceq.lm.Attention("softmax_x", 4, 1).operator` and
`ceq.bench._softmax_operator` is not used anywhere in this node.** `bench` masks
with `tril(-1)`: strictly causal, diagonal excluded, so `A_ii = 0` and row 0 is
empty, while this arm needs `P_ii = 1` for the label. ARM PL's node found that
running the parity bind against `bench` as a primary reference puts two binds on
two different objects, which is V-24's shared-carrier defect. It is not repeated.

| claim | result |
|---|---|
| `torch.equal(op.real, lm softmax_x)` | **True** (bitwise) |
| `torch.equal(op.imag, 0)` | **True** |
| `torch.equal(op.real @ v, ref @ v)` | **True** (bitwise) |
| the same bind on the shipped `ArmPhase` module at `identity_heads()` | **True** |
| `readout(...)` through a **complex** gemm | **`1.1102230246251565e-16` — NOT bitwise** |

**What the bind says, in V-24's words and no wider:** *the modification enters
only through the key logit, additively, and through a unimodular twist, and both
vanish at that setting.* It is not "bitwise standard attention" unqualified —
that is §S-M's original clause and `CEQ.V15.gate_zero_not_stochastic` refuted it.
The module bind is run because a parity bind on a free function the shipped
module does not dispatch through is a reading of a non-shipped operator.

> **PLANTED NEGATIVE 4** — one mutilation per head, from the identity setting:
>
> | moved head | `max \|op − ref\|` |
> |---|---|
> | `m` only (random on `[0, 0.8]`) | **`0.562760`** |
> | `θ` only (standard normal) | **`0.339569`** |
> | `s` only (standard normal) | **`0.478112`** |
>
> and the class-closure check at a random `θ`: the modulus row still sums to `1`
> to `< 1e-14` and is strictly positive on the causal triangle, so the family
> never leaves the softmax class between the identity point and anywhere else.

---

## 6. BIND 5 — THE CHAIN PATH-PRODUCT LABEL, AND WHAT IT COSTS

**On the open interior it holds at machine precision.** `|a|` uniform on
`(0.15, 0.85)`, `arg a` uniform on `(−π, π)`, drives standard normal — the fork
probe's draw made complex:

| shape | residual | bar |
|---|---|---|
| `s = 8` | **`9.155133597044475e-16`** | `1e-6` |
| `s = 64` | **`5.2510145522368515e-15`** | `1e-6` |

ARM PL's published figure is `6.6613381477509392e-16` at `s = 8`. **These are
not the same number and are not meant to be** — ARM PL reproduces a real chain,
this reproduces a complex one on a different draw. The comparable statement is
that both land at machine precision, nine orders inside the bar, and that the
`s = 8 → s = 64` growth (`5.7×` here, `11×` for ARM PL) is telescoping round-off
in both.

**On BED-M's band it does not hold, and this is the delta's cost.**

```
  band draw, |a| = 1, theta in {0, pi}:
    residual = nan     v_max = inf     1/(1-m) = inf
    s_j = [0.0, -inf, -inf, ...]       V_j = [0.0, inf, -inf, ...]
```

The residual is `nan` rather than large: `V_j = b_j/(1 − 1) = ±inf` meets a
softmax weight of `0` from the `−inf` key bias. **The test asserts
`not (residual <= 1e-6)` and `isnan`, so a `nan` cannot be read as a pass in
either direction.**

**The ladder in from the band prices it, and locates the cost exactly.**

| `m = 1 − ε` | residual | `v_max` | `1/(1−m)` |
|---|---|---|---|
| `ε = 1e-2` | `1.332268e-15` | `2.094085e+02` | `1.000000e+02` |
| `ε = 1e-4` | `4.440892e-15` | `2.094085e+04` | `1.000000e+04` |
| `ε = 1e-6` | `2.220446e-15` | `2.094085e+06` | `1.000000e+06` |
| `ε = 1e-9` | `2.442491e-15` | `2.094085e+09` | `1.000000e+09` |
| `ε = 1e-12` | `6.883383e-15` | `2.094131e+12` | `1.000022e+12` |
| **`ε = 0`** | **`nan`** | **`inf`** | **`inf`** |

**The bind survives to `1 − 1e-12` at machine precision and dies only AT the
endpoint.** So the reading is not "the phase arm cannot do the label". It is:
**the cap closed the GATE half of `V15_R1.md` §1's hole and left the VALUE half
exactly where `V15_JUPITER2_FORK.md` §8.4 cost 6 put it.** R1 §1 measured ARM PL's
oracle setting evaluated on BED-M's own coefficients:

```
  a = +1 :  g = log a = 0.0      s = log1p(-a) = -inf      V = b/(1-a) = inf
  a = -1 :  g = log a = nan      s = log1p(-a) = 0.6931    V = 0.5
  a =  0 :  g = log a = -inf     s = 0.0                   V = 1.0
```

Under ARM PHASE the gate is exactly representable at all three — `m ∈ {0, 1}`,
`θ ∈ {0, π}`, no `nan` — which is a real change: **the `a = −1` cell, where ARM
PL's `log a` is `nan`, is now `m = 1, θ = π`, an ordinary point of the
parametrization.** The `a = +1` value column is `inf` in both. **Half of R1 §1's
limit is removed and half is not, and the half that is not is the half the label
bind needs.**

> **PLANTED NEGATIVE 5** — five mutilations of the (L) setting, all firing at
> `O(1)`, with ARM PL's own four beside them on its own draw so the two arms'
> rejection regions can be compared rather than asserted equivalent:
>
> | mutilation | ARM PHASE | ARM PL (`V15_ARM_PL.md`) |
> |---|---|---|
> | `none` (the honest cell) | `0.000000` | `0.000000` |
> | `drop_key_bias` | **`0.838865`** | `0.974946` |
> | `drop_value_rescale` | **`0.889267`** | `0.916527` |
> | `drop_bos_sink` | **`1.000000`** | `1.000000` |
> | `half_key_bias` | **`0.423865`** | `0.484493` |
> | `drop_phase` (this arm's own) | **`0.933220`** | — no analogue |
>
> `drop_phase` is the one that is new: deleting the twist leaves a
> magnitude-only chain, and no real chain reproduces the complex label. The
> figures differ from ARM PL's because the draw is complex, not because the
> mutilations differ; the mutilation names and semantics are identical.
>
> The identity manifest moves under **all five**: `label_cell` folds a
> `PHASE_FIELDS` block (`variant`, three head settings, `bos_value`, `m_max`,
> `v_max`, `dyn_range_bound`) into the base hash, so a mutilated cell cannot be
> filed under an honest cell's hash.

---

## 7. WHAT THE PHASE CONSTRUCTION GIVES UP RELATIVE TO ARM PL

Seven things, matching §8.4's own form, none rhetorical.

1. **One more scalar head, `+17` parameters** (`4,820` against `4,803`). It is
   not free slack and it cannot be saved: tying `s = log(1 − m)` would remove
   the head, but that tie is `−inf` at `m = 1`, which is **precisely the
   identity setting**, so the saving would delete the parity bind.

2. **The read-out is no longer bitwise.** The **operator** is
   (`torch.equal` = True). Contracting it against `V` goes through a complex
   gemm, a different reduction than the real one, and costs
   **`1.1102230246251565e-16`**. `op.real @ v` *is* bitwise, so the loss is in
   the complex contraction and not in the construction — but any future claim
   must name which tensor it means.

3. **The complex row does not sum to `1`.** Only `|A_i·|` does. ARM PL's row was
   a genuine probability vector end to end; here the probability statement is
   about the modulus and the twist is a separate carrier. Two quantities where
   there was one.

4. **The identity point moved from "all heads zero" to `m = 1, θ = 0, s = 0`.**
   The magnitude head's **bias is one**, not zero — hence `identity_heads()`
   rather than `zero_heads()`. **Any harness that resets heads to zero puts this
   arm at `m = 0`, the annihilating gate, not at the identity.** That is a
   live trap for a drop-in and it is why the method was renamed.

5. **`m = 0` is now attainable from a whole half-line, and prefix-PHASE is
   undefined past it.** Once a magnitude is `0`, `Re C = −inf` at every later
   position and `C_i − C_j` is `nan` for two of them; `cumprod` returns the true
   `0`. ARM PL has the identical hole at `a = 0` (`g = log 0 = −inf`), so this
   is **inherited, not added** — but ARM PL reached it only at the single
   parameter value `g = −inf`, and the cap reaches it from every `u ≤ 0`.
   `test_the_prefix_route_is_undefined_once_a_magnitude_hits_zero` is the
   receipt.

6. **The label bind still dies on the band** (§6). §8.4 cost 6 is unmoved. The
   delta's `|a| ≤ 1` claim is about the **gate**; the value dynamic range
   `1/(1 − m)` is untouched, and the ladder in §6 shows it reaching `1.0e+12`
   before the endpoint.

7. **A hard cap trades a divergence for a DEAD ZONE, and that is new.**
   `d(clamp)/du` measured:

   ```
     u      -5.0   -1e-9    0.0    0.5    1.0   1+1e-9    2.0   285.0719
     m       0.0     0.0    0.0    0.5    1.0      1.0    1.0        1.0
     dm/du   0.0     0.0    1.0    1.0    1.0      0.0    0.0        0.0
   ```

   **A magnitude driven above `1` receives zero gradient through the magnitude
   path and cannot be pushed back in.** ARM PL's `â = exp(g)` is differentiable
   everywhere, so its `â_max = 285.07` seed was at least still connected to its
   loss. This is a mechanism note filed against the delta's own
   pre-registration, **not a second prediction** (M-20): §X₃₆ registers
   *"8/8 converge"* and its author's counter registers *"crossing shrinks to
   ≤ 3/8"*, and both are already filed and scored. The observation here is that
   the counter now has a **named mechanism inside the construction** —
   saturation with no gradient — rather than only *"the cap removes gain"*.

---

## 8. THE DIAGNOSTIC, RUN ON THE CORPUS ALONE AND AT ZERO STEPS

`V15_R1.md` §6 found that M-18's *prescribed replacement* was itself
non-discriminating: the trained `sign(a)` probe read `p = 0.917953` against a
**zero-step control of `0.943741` — higher**. The target was so easy the
predictor saturated before training. So **no diagnostic is trusted here until
both controls are read.** Nothing is trained; only the two controls exist at
this node.

Instrument: the gate `R²`, `Re(â)` against the corpus's `a` on the live band, by
a least-squares probe **fit on the train batch and scored on the eval batch**
(different seed, different tensor) — `scripts/v15_r1.py::probe`'s form. BED-M
`e3_t2`, `s = 64`, `d = 24`, `d_model = 16`, live band `{62, 63}`.

| reading | `n_tr=512, n_ev=512` | `n_tr=2048, n_ev=4096` (R1's shape) |
|---|---|---|
| **(a) corpus alone, all 16 channels, NO arm** | **`1.000000`** | **`1.000000`** |
| **(a) corpus alone, `CH_DRIVE` only** | `1.000000` | `1.000000` |
| **(b) zero-step `ArmPhase`, 8 seeds** | `0.291617` | **`0.300689`** |
| **(b) zero-step `ArmPL`, 8 seeds, same process** | `0.356876` | **`0.371583`** |
| `Var(a)` on the live band | `1.000668` | `1.000079` |
| `\|a\|` values present on the band | `{1.0}` | `{1.0}` |

Per-seed zero-step `ArmPhase` at R1's shape: `0.207931, 0.153880, 0.257340,
0.596127, 0.097731, 0.340202, 0.099744, 0.652558`.

**Three readings.**

1. **`R² = 1.000000` on the corpus alone confirms M-18's mechanism in its
   sharpest form: the diagnostic's target IS an input channel.**
   `scale/negation_scope.py` writes `x[:, :, CH_DRIVE] = a`, so a linear probe
   on the raw corpus recovers `a` exactly. Any gate `R²` short of `1.0` is
   therefore measuring the **arm's parametrization**, never the availability of
   the information. The `V15_SATURN2_LEAK_RULING.md` correction inside M-18 says
   the same thing from the other side, and this is its numeric restatement on
   this node's own draw.

2. **The instrument nevertheless has real headroom, which is exactly what the
   `sign(a)` probe lacked.** Floor `0.30`–`0.37` at zero steps, R1's trained
   reading `0.699299`, ceiling `1.000000`. The `sign(a)` probe's floor was
   `0.943741` against a ceiling of `1.0` — `0.056` of room, and R1 measured a
   *negative* trained gain of `−0.025787` inside it. **The gate `R²` is carried
   forward and the `sign(a)` probe is not registered by this node at all.**

3. **The ARM PL zero-step control reproduces R1's published value.**
   `0.371583` here against `0.371839` in `V15_R1.md` §6 — a difference of
   `2.56e-4`, inside M-10's `2.345e-3` cross-thread reduction-order floor. R1
   pinned `threads = 8`; this box ran at `20`. Same instrument, same corpus,
   same seeding plan.

4. **`ArmPhase`'s zero-step reading is `0.0709` BELOW `ArmPL`'s** (`0.300689`
   vs `0.371583`, R1's shape, all eight seeds in the same process). This is an
   **initialization** property, not a capability statement: the cap sends every
   `u < 0` to exactly `0`, which flattens roughly half of a zero-mean head's
   output onto a constant and removes its contribution to the probe's variance.
   It is reported because a re-run comparing trained gate `R²` across the two
   arms must subtract each arm's own zero-step baseline, not a shared one.

---

## 9. REGRESSION

```
$ python -m pytest tests/arm_phase -q
30 passed in 1.79s

$ python -m pytest tests/arm_pl -q
17 passed in 2.08s

$ python -m pytest tests/arm_pl tests/loop -q --no-header -p no:cacheprovider 2>&1 | tail -5
FAILED tests/loop/test_every_boundary_node_can_propagate.py::...[reproduce-n1024-d4-t0.95]
FAILED tests/loop/test_manifest_refuses_an_absence_it_has_not_earned.py::...
FAILED tests/loop/test_the_bar_control_is_scored_out_of_sample.py::...
15 failed, 535 passed, 3 warnings in 34.79s
```

`535 = 518 + 17`, the two suites' own counts.

The `tests/loop` baseline was measured **A/B in the same session**, this node's
two paths moved out to a scratch directory and back:

```
--- WITHOUT this node's files ---
15 failed, 517 passed, 3 warnings in 31.22s
--- WITH this node's files ---
15 failed, 518 passed, 3 warnings in 31.37s
```

`15 failed` in both, and the fifteen are the same fifteen by name (ten
`test_no_test_file_imports_conftest_as_a_bare_module` parametrizations over
`tests/chase/` and `attic/tests/chase/`, two
`test_corpus_is_recoverable_and_verifiable`, one
`test_no_corpus_instance_has_a_boundary_node_that_cannot_propagate[reproduce-n1024-d4-t0.95]`,
one `test_no_weight_record_omits_a_declared_identity_field`, one
`test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on`). The `+1`
passing case is
`test_no_module_writes_a_file_at_import.py::test_module_has_no_import_time_write_or_argv_read[ceq/arm_phase.py]`,
a parametrization over `ceq/*.py` that the new module joins and passes.

**The brief's standing baseline of `15 failed / 515 passed` is stale at this
HEAD.** Measured here without this node's files it is **`15 failed / 517
passed`**, twice, deterministically. The failure count and names are what the
brief's check is about and both are unchanged; the passed count has grown by two
between the commit that recorded `515` and `bce1556`. This is reported rather
than quietly matched.

---

## LIMITS

- **Nothing here is trained.** Every number in this file is a float64/complex128
  identity residual, a bitwise comparison, or a zero-step control. The delta's
  pre-registered `8/8 converge` and its counter-prediction are **not** tested by
  this node and are not commented on beyond §7 item 7.
- **The label bind's oracle setting still does not cover BED-M.** Its hypothesis
  is `|a| < 1`; BED-M supplies `|a| ∈ {0, 1}`. The gate half of that gap is
  closed by the cap and the value half is not (§6). Any claim that ARM PHASE
  "expresses the chain" is an expressibility statement about a magnitude range
  this cell would never train on — the identical caveat `V15_R1.md` LIMITS
  records for ARM PL, unchanged in force.
- **The gate `R²` instrument was not pre-registered by anyone.** R1 found it
  post hoc. It is carried forward here with both controls read, which is more
  than the `sign(a)` probe had, but it remains an observation-grade instrument
  and this node does not promote it to a kill condition.
- **`ArmPhase`'s zero-step gate `R²` is below `ArmPL`'s by `0.0709`.** The
  mechanism offered in §8 item 4 (half the head flattened onto the cap's lower
  endpoint) is an explanation this node did not test against an alternative.
- **The prior art is one item deep.** Only LRU got an equation with a numeric
  instance here (§2). Unitary/orthogonal RNNs, complex-diagonal S4/Mamba and
  RoPE are named in `CEQ_V15_3_DELTA.md` and are **not** `[V-eq]`; no claim in
  this file cites them.
- **`X₃₇`'s (b) persistent `β₁` and (c) Euler–Poincaré certificates are not
  built here.** Only (a), the integer winding, is — and only as a bind on the
  parity mask, not as a per-instance certificate over a learned trajectory.
- **Lean #16 `unit_phase_product` is not written.** §3 is its float64 shadow at
  `1e4` phases; the theorem itself is another node's.
- **The complex read-out costs one ulp** (§7 item 2). Every residual in §6 is
  measured through that path, so each carries it; at `9.16e-16` against a
  `1.11e-16` floor the bind is roughly eight ulps of headroom, not nine orders.
  The nine-orders figure is against the contract's `1e-6` bar, which is the
  comparison the contract asks for, and both are stated.

---

**Distance to the north star.** `|a| ≤ 1` is now a property of the arithmetic:
2,200,000 draws spanning ten orders of magnitude on either side of the cap, worst
`|a| = 1.0`, including the exact gate magnitudes that put three of R1's eight
seeds outside the construction's domain. The parity half gains an exact
statement it did not have — `θ ∈ {0, π}` reproduces `V15.lean`'s signed mask on
`4096/4096` entries with `torch.equal` — and the standard-attention half is
bitwise at the operator against the repo's own inclusive-causal control. What is
**not** bought is the value path: on BED-M's band the label bind reads `nan`,
and the ladder shows `1/(1 − m)` reaching `1.0e+12` one step before it. **The
claim sentence these binds permit is: the gate can no longer leave its
admissible range, and the value rescale can still leave its own.**
