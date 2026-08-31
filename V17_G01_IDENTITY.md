# V17-K GATE-0 — G0.1, THE FOUR IDENTITY BINDS, RE-MEASURED

Discharges `CEQ_V16_CONTRACT.md`'s v17-K Gate-0 item **G0.1**:

> identity binds under AD-1: oracle gates ⇒ label ≤1e-6; softmax corner row
> sums 1.000000; |a| ≤ 1, zero exceedances; deterministic-mode assertion PASS.

**Files written:** `tests/gate0/test_g01_identity.py` (new), this file. Nothing
else. `tests/gate0/conftest.py` was created by this node and then replaced by a
parallel node with a richer shared version at 21:29; the suite here does not
import it and passes either way. No file under `ceq/`, `scale/`, `scripts/`,
`lean/`, `MISTAKES.md` or `CEQ_V16_CONTRACT.md` was touched. **No git command
that writes was run.** No kaggle CLI call, no upload, no touch of `~/.kaggle`.

**Provenance.** `git rev-parse HEAD` = `ab5b48547884e04258276e6e808d5a71ea65f917`
at the start of this node's work **and** at the end (no commit was made by
anyone during it). `git status --porcelain` at start: **clean**. At end:
`M ceq/hf/train.py`, `?? ceq/kdata.py`, `?? kaggle/`, `?? requirements-kaggle.txt`,
`?? results/k_data_manifest.json`, `?? scripts/k_cert.py`, `?? tests/gate0/` —
every one of those except `tests/gate0/test_g01_identity.py` belongs to one of
the four parallel Gate-0 nodes, not to this one.

**Box** `[MEASURED]`: Windows 11, torch `2.5.1+cu121`, `torch.get_num_threads()
= 20`, one `NVIDIA GeForce RTX 4060 Laptop GPU`. Binds A/B/C on **cpu at
float64/complex128**; bind D on **cuda**, in a subprocess, at `s = 64` — the
smallest shape that instantiates the kernel in question. **Nothing was
trained**; no NRMSE, no cell of any bed, no seed pooling appears below.

**Nothing is inherited.** `V16_ARM_SMPRIME.md` (a) reports `5.919777e-16` and
`7.550528e-16`. Those were not carried forward — every number in this file was
computed by the run whose output is pasted in §2, and the fact that two of them
land on the same digits is a **reproduction**, reported as such.

---

## LEAD CAVEAT — read this or get the item backwards

**"Deterministic-mode assertion PASS" is true of the FORWARD path only, and the
BACKWARD does not run.** `arm_smprime` is built on `cumprod` precisely because
`V16_R1_DEVICE_READY.md` found `cumsum_cuda_kernel` has no deterministic
implementation. Measured on this box this run: **`torch.cumprod` forward is
deterministic-clean on cuda (float64 and complex128), and `torch.cumprod`'s
BACKWARD is not — it calls `cumsum_cuda_kernel` and raises the identical
error.** So under `torch.use_deterministic_algorithms(True)`:

| what | on cuda, strict determinism |
|---|---|
| the (L) read-out, `label_cell`, `path_product` | **RUNS**, bitwise identical across two separate processes |
| `ArmSMPrime.forward` under `no_grad` | **RUNS**, bitwise identical across two separate processes |
| `ArmSMPrime` forward **+ backward** | **RAISES** `cumsum_cuda_kernel does not have a deterministic implementation` |

G0.1's four binds are all forward-path statements, so G0.1 is **GREEN**. But a
reader who stops at "PASS" and schedules a Kaggle **training** run under strict
determinism will get a `RuntimeError` on the first `.backward()`. That run needs
`warn_only=True`, which was measured here and does work (§5). The suggestion in
`V16_R1_DEVICE_READY.md` §8 that `cumprod` is the deterministic replacement for
`cumsum` is **half right and the wrong half is the training half**.

Two further things a fast reader would get wrong, both established below:

- **Bind C is a property of `clamp`, not an empirical finding.** `m ≤ 1` cannot
  fail while the cap is applied last and the input is finite. Its content is
  entirely in the three things that *can* fail: the rejected cap order, a `nan`
  magnitude, and the counter used to read it. All three are exercised (§4).
- **`Σ_j |W_ij| = 1.000000` at `β = 1` is a VACUOUS statistic for the `g`
  switch.** `V16_ARM_SMPRIME.md` (e) reads the softmax corner with it. It is an
  identity at `β = 1` for *every* gate whatsoever, measured at `4.440892e-16`
  with the gates fully on. The corner is carried by `Σ_j Re W_ij`, which reads
  `1.750255` at `g = 1` (§3).

---

## VERDICT TABLE

| bind | measured, this box, this run | bar | verdict |
|---|---|---|---|
| **A. oracle gates ⇒ label** | `5.919777e-16` at `n=64,s=64`; `7.550528e-16` at `n=512,s=64`; `7.550528e-16` at `n=256,s=128`. Support `{−1,0,+1}` asserted on every draw; `3968/4096`, `31744/32768`, `32256/32768` gates exactly zero; label `y.std` `1.494263 / 1.432995 / 1.468677` | `1e-6` | **GREEN**, ten orders inside |
| **B. softmax corner row sums** | `max_i \|Σ_j Re W_ij − 1\| = 2.220446e-16` over `64/64` rows; `41/64` rows bitwise `1.0`; `0` rows outside `5e-7` | `1.000000` | **GREEN** |
| **C. \|a\| ≤ 1, zero exceedances** | `0` exceedances on a `13×8` adversarial `(u,g)` sweep and on its induced hop products; `0` on BED-M's own `\|a\|`; `0` on the shipped `ArmSMPrime` across 8 seeds, `â_max = 1.000000` each. Both closed endpoints attained: `m` range `[0.0, 1.0]`, BED-M `\|a\|` `= 0` on 3968 and `= 1` on 128 entries | `≤ 1`, zero | **GREEN** |
| **D. deterministic-mode assertion** | flag live (proved by `cumsum` raising), `CUBLAS_WORKSPACE_CONFIG=:4096:8`; read-out bitwise identical **within** a process and **across two processes**, sha `e547261108d1fb83…`; `ArmSMPrime.forward` likewise, sha `0f58a4f19d1519b5…`; `label_cell` residual on cuda `3.552714e-15` | runs + bitwise | **GREEN on forward. The backward RAISES** — see LEAD CAVEAT |
| **G0.1 overall** | | | **GREEN** |

**Reproduction, not inheritance:** bind A's three residuals land on the same
digits `V16_ARM_SMPRIME.md` (a) published against a different HEAD
(`bbbc688…`). Recomputed here at `ab5b485…`.

---

## 1. RED

Tests were written first. The four binds hold on the tree as it stands, so
there is no version of this suite that goes RED by the arm being wrong. The
RED that matters is therefore the one this item's own rule demands: **the
planted negatives failing to be caught by a deliberately weakened assertion.**
`G01_WEAKEN=1` replaces each bar with a vacuous one and each exceedance counter
with a naive one-sided, `nan`-blind form. Under it seven tests fail — and every
one of them is a *negative that stopped firing* or a *comparator that went
blind*, which is exactly the failure mode that put 14 vacuous controls in
`STRUCK.md`.

```
$ G01_WEAKEN=1 python -m pytest tests/gate0/test_g01_identity.py -q

E   AssertionError: {1e-12: 4.277467301349765e-12, 1e-09: 4.277340259761818e-09,
                     1e-06: 4.277340293512566e-06, 0.001: 0.004277340292930809}
E   assert not 0.004277340292930809 <= 1000000.0

E   AssertionError: drop_phase did not fire: 8.554680585861206
E   assert not 8.554680585861206 <= 1000000.0

E   AssertionError: drop_magnitude did not fire: 19.80139898136258
E   assert not 19.80139898136258 <= 1000000.0

E   AssertionError: beta_one did not fire: 2.851560195287069
E   assert not 2.851560195287069 <= 1000000.0

E   AssertionError: beta=0.999999 did not fire: 4.9325420385937235e-06
E   assert not 4.9325420385937235e-06 <= 1.0

E   AssertionError: the rejected cap order did not fire: 0
E   assert 0 > 0

E   AssertionError: the comparator is blind
E   assert 8.964 != 8.964

FAILED tests/gate0/test_g01_identity.py::test_bind_a_the_residual_fires_on_a_perturbed_gate
FAILED tests/gate0/test_g01_identity.py::test_bind_a_planted_negatives_fire_on_the_real_corpus[drop_phase]
FAILED tests/gate0/test_g01_identity.py::test_bind_a_planted_negatives_fire_on_the_real_corpus[drop_magnitude]
FAILED tests/gate0/test_g01_identity.py::test_bind_a_planted_negatives_fire_on_the_real_corpus[beta_one]
FAILED tests/gate0/test_g01_identity.py::test_bind_b_the_row_sum_fires_off_the_corner[0.999999]
FAILED tests/gate0/test_g01_identity.py::test_bind_c_the_counter_fires_on_the_wrong_cap_order_and_on_nan
FAILED tests/gate0/test_g01_identity.py::test_bind_d_deterministic_mode_on_cuda
7 failed, 19 passed in 9.57s
```

**Two of those seven were live RED against the strict bar first, and both were
instrument defects rather than deliberate weakenings.** Recorded because which
is which is the whole value of writing the tests first:

| what went RED | what it found | resolution |
|---|---|---|
| `..._planted_negatives_fire_on_the_real_corpus` — **all four**, on the *honest* cell: `AssertionError: 1.4404945373535156` | The negatives were being read through `ceq/arm_smprime.py::label_cell`, which scores **every** position against `chain_label`. The honest cell reads `1.44` on BED-M's real corpus, not `1e-16` | **instrument changed**, not the arm: the negatives are now read on the SAME comparator the bind is (last position, against the bed's own oracle) by threading `smp.mutate` through it. The `1.44` is a real defect in a file this node does not own and it is filed in §6 with a test that pins its mechanism |
| `..._row_sum_fires_off_the_g_corner`: `g=1 did not fire: 4.440892098500626e-16` | `Σ_j \|W_ij\|` is an **identity** at `β = 1` and cannot see the `g` switch at all. The intended planted negative was unfireable — an empty rejection region | **claim corrected to what is true.** The test now measures both statistics, asserts the absolute row sum is `≤ 1e-12` (i.e. asserts it VACUOUS, so no later node can cite it as a corner certificate) and reads the corner on `Σ_j Re W_ij` |

## 1b. GREEN

```
$ python -m pytest tests/gate0/test_g01_identity.py -q -s

  BIND A  n=  64 s= 64  residual = 5.919777e-16   bar 1e-06   support [-1.0, 0.0, 1.0]  zeros 3968/4096  y.std 1.494263
  BIND A  n= 512 s= 64  residual = 7.550528e-16   bar 1e-06   support [-1.0, 0.0, 1.0]  zeros 31744/32768  y.std 1.432995
  BIND A  n= 256 s=128  residual = 7.550528e-16   bar 1e-06   support [-1.0, 0.0, 1.0]  zeros 32256/32768  y.std 1.468677
  BIND A  perturbation ladder (gate + eps, label held): 1e-12->4.277467e-12  1e-09->4.277340e-09  1e-06->4.277340e-06  1e-03->4.277340e-03
  BIND A  negative drop_phase       residual = 8.554681e+00   (honest 5.919777e-16)
  BIND A  negative drop_magnitude   residual = 1.980140e+01   (honest 5.919777e-16)
  BIND A  negative beta_one         residual = 2.851560e+00   (honest 5.919777e-16)
  BIND A  negative exp_scan         residual = nan            (honest 5.919777e-16)
  BIND A  label_cell on the real corpus = 1.440495e+00   max|b_0| = 1.440495e+00   with b_0 zeroed = 2.965914e-16   last-position bind = 2.965914e-16
  BIND B  rows 64  max|sum-1| = 2.220446e-16   bitwise-exact rows 41/64   out-of-bar 0   last-row max/min 20.858   (|W| row sum dev 2.220446e-16 -- vacuous, see the g-corner test)
  BIND B  negative beta=0.0       max|sum-1| = 1.377300e+02
  BIND B  negative beta=0.5       max|sum-1| = 1.077837e+01
  BIND B  negative beta=0.999999  max|sum-1| = 4.932542e-06
  BIND B  negative g=1.0   max|sum|W|-1| = 4.440892e-16  (VACUOUS: an identity at beta=1)   max|sum Re W -1| = 1.750255e+00
  BIND C  sweep 13x8  exceedances = 0   uncapped-control exceedances = 61   m range [0.0, 1.0]
  BIND C  BED-M |a|: min 0.0  max 1.0   at 0: 3968   at 1: 128   exceedances 0
  BIND C  ArmSMPrime seed 0..7  a_hat_max = 1.000000   min 0.000000   exceedances 0     [all eight identical]
  BIND C  negative cap-first-blend-second: exceedances = 35   range [-999999.0, 3.0] | nan magnitude: counted = 1
  BIND D  { ... full JSON in section 5 ... }
26 passed in 8.59s
```

Run **twice**; every printed number is identical between the two runs
`[MEASURED]`. Wall clock `8.59 s` and `6.15 s` for the 26 tests including two
cuda subprocess launches — `[MEASURED]`, this box, this run, nothing inherited
(L-TIME).

**Sibling regression:** `python -m pytest tests/gate0` (all five Gate-0 nodes'
files together) reports **no** failures at the time of the final run. Earlier in
this node's work the same command was interrupted at collection by
`tests/gate0/test_g06_kcert.py`'s `ImportError: cannot import name 'k_cert'`,
which was that node's own RED and has since gone green.

---

## 2. BIND A — oracle gates ⇒ label residual ≤ 1e-6

**Setting.** `ceq/arm_smprime.py`'s (L) corner: `β = 0`, QK off,
`(u, θ) = (|a|, arg a)`, `V = b` **unrescaled**. `#5a`'s third corner and `#2`
re-stated's clauses 1 and 4 evaluated directly. No `log(1−m)` and no `1/(1−m)`
is instantiated anywhere on this path, which is what makes `m = 1` an ordinary
point rather than the `nan` `V15_ARM_PHASE.md` (e) reports.

**Corpus.** `scale/negation_scope.py::make_equilibrium_batch(n, s, 24,
t_star=2, d_model=16, seed=0)`, cast to float64. **The label is recomputed by
the bed's own `equilibrium_oracle` at float64**, so what is measured is the
arm's residual and not the corpus's float32 storage. Read at the **last
position**, which is where `equilibrium_oracle` is defined.

**L-DOM census, asserted on every draw before the residual is read**
`[MEASURED]`:

| shape | gate support | gates exactly `0` | label `std` |
|---|---|---|---|
| `n=64, s=64` | `[−1.0, 0.0, 1.0]` | `3968 / 4096` | `1.494263` |
| `n=512, s=64` | `[−1.0, 0.0, 1.0]` | `31744 / 32768` | `1.432995` |
| `n=256, s=128` | `[−1.0, 0.0, 1.0]` | `32256 / 32768` | `1.468677` |

**Residuals** `[MEASURED]`: `5.919777e-16`, `7.550528e-16`, `7.550528e-16`.
Bar `1e-6`. All three **ten orders inside**.

### Planted negatives — four, all fire

Read through `ceq/arm_smprime.py::mutate` as shipped, on the **real BED-M
corpus** at `n=64, s=64`, against the **same comparator** the bind is read on:

| mutation | what it deletes | residual `[MEASURED]` |
|---|---|---|
| `drop_phase` | the sign carrier | `8.554681e+00` |
| `drop_magnitude` | the annihilator | `1.980140e+01` |
| `beta_one` | normalizes the row the path product weights | `2.851560e+00` |
| `exp_scan` | takes `exp(C_i − C_j)`, the route `no_prefix_scan_represents_a_zero_gate` forbids | **`nan`** |
| honest | — | `5.919777e-16` |

`exp_scan` reading `nan` rather than a large number is the theorem, not an
accident: `−inf − (−inf)` is what a prefix scan does at a zero magnitude, and
BED-M puts a zero on `3968/4096` of the gates. The rejection region is occupied
by **shipped code** (`ROUTES = ("product", "exp_scan")` lives in the arm), not
by a test-local copy — `MISTAKES.md` V-24.

### Non-degeneracy of the PASS half

1. **The target is not degenerate.** Asserted, not assumed:
   `y.abs().max() > 0` and `y.std() > 0.1` on every draw, measured at
   `1.43`–`1.49`. A residual against an all-zero or constant label is a PASS on
   a degenerate input and is void.
2. **The draw exercises clause 4.** `(a == 0).sum() > 0` asserted; measured at
   `3968/4096` and up. Without a zero gate the whole reason this arm is a path
   product and not a scan is untested.
3. **The comparator would have caught a perturbation.** The oracle gates are
   moved off truth by `ε` at the last hop, **with the bed's label held fixed**,
   and the residual is read again `[MEASURED]`:

   | `ε` | residual | verdict at bar `1e-6` |
   |---|---|---|
   | `1e-12` | `4.277467e-12` | inside — as it must be |
   | `1e-9` | `4.277340e-09` | inside |
   | `1e-6` | `4.277340e-06` | **fires** |
   | `1e-3` | `4.277340e-03` | **fires** |

   The response is linear in `ε` with gain `4.2773`, so the instrument's
   sensitivity is a measured number and not a hope: **the bind's `1e-6` bar
   corresponds to a gate error of `2.34e-7`.** Under `G01_WEAKEN` this ladder
   is the first thing that goes RED.

---

## 3. BIND B — softmax corner row sums 1.000000

**Setting.** `β = 1`, `g ≡ 0`, QK on — `#5a`'s first corner, reached through
the gate rather than around it. The corner is verified as a **setting of the
switches** before it is used: `blend(u, θ, 0.0)` is asserted to return `m = 1`
and `θ_eff = 0` **bitwise** (`torch.equal`), so this is the `g == 0` corner and
not something near it. The operator's imaginary part is then asserted **exactly
`0.0`**, so the row sum is a real quantity.

`[MEASURED]`, `n = 64` rows, `d = 8`, float64:

| quantity | value |
|---|---|
| `max_i \|Σ_j Re W_ij − 1\|` | `2.220446e-16` |
| rows bitwise equal to `1.0` | `41 / 64` |
| rows outside `5e-7` | `0 / 64` |
| last row `max_j W / min_j W` | `20.858` |

**On "exact".** The item says `1.000000`; the bar enforced is `5e-7`, i.e. the
deviation rounds to `1.000000` at six places. `41/64` rows are **bitwise**
`1.0`; the other 23 are one ulp off. Asserting bitwise `1.0` on all 64 would be
a bar this arithmetic cannot meet and does not need to — `Σ_j x_j/Z` is not
`(Σ_j x_j)/Z` in float64 — so the exact count is reported as a **measured
fact** and the bar is the contract's stated precision. That distinction is
named rather than papered over.

### Planted negatives — the `β` switch

| negative | `max_i \|Σ_j Re W_ij − 1\|` `[MEASURED]` | fires at `5e-7`? |
|---|---|---|
| `β = 0` (linear-attention corner) | `1.377300e+02` | yes |
| `β = 0.5` | `1.077837e+01` | yes |
| `β = 0.999999` | `4.932542e-06` | **yes** |

The `0.999999` rung is what says the bar is `5e-7` and not `1e-2`: a
**one-part-in-a-million** move of the switch is caught with an order of margin.
Under `G01_WEAKEN` (bar `1.0`) it is the rung that stops firing.

### The `g` switch, and the statistic that cannot see it

**This is a live correction to `V16_ARM_SMPRIME.md` (e).** That table certifies
the softmax corner with `Σ_j |W_ij| = 1.000000` at `β = 1`. Measured here with
the gates **fully on** (`g = 1`, drawn magnitudes and phases):

| statistic at `β = 1, g = 1` | `[MEASURED]` |
|---|---|
| `max_i \| Σ_j \|W_ij\| − 1 \|` | **`4.440892e-16`** — still one |
| `max_i \| Σ_j Re W_ij − 1 \|` | **`1.750255e+00`** — off the corner |

The reason is structural and it is in the arm's own construction:
`|W_ij| = R_ij e_ij / Z_i` and `Z_i = Σ_j R_ij e_ij` **by definition**, so the
absolute row sum is exactly `1` at `β = 1` for *every* gate, every `g`, and
every phase. It certifies the `β` switch and **nothing else**; used as a corner
certificate it is a control that cannot fail. The suite asserts it `≤ 1e-12`
— i.e. **asserts it vacuous** — so a later node reading this file cannot cite
it as evidence about `g`.

### Non-degeneracy of the PASS half

A row of equal weights sums to `1` for a reason that has nothing to do with the
normalizer, so the check is asserted to be reading a real softmax: the last
row's `max/min` ratio is asserted `> 10` and measured at **`20.858`**. Row `0`
is trivially `1` at this corner (one term); the reading is over all 64 rows and
the maximum deviation is taken, so the trivial row cannot carry it.

---

## 4. BIND C — |a| ≤ 1 under AD-1's closed [0,1], zero exceedances

**AD-1 as realised.** `m = clamp(lerp(1, u, g), 0, 1)`, **cap last**. AD-1 is
signed and fixed at the closed interval, and `0` and `1` are attainable
**values**.

**Say plainly what this bind is.** With the cap applied last and a finite
input, `m ∈ [0,1]` is a property of `clamp` and **cannot fail**. A PASS on it
is worth nothing on its own. Its entire content is in the three things that
*can* fail and that are exercised here: the **cap order**, a **`nan`**
magnitude, and the **counter** used to read it.

**The counter.** `n_outside = |{ t : ¬(t ≥ 0 ∧ t ≤ 1) }|` — two-sided and
`nan`-safe. `nan ≥ 0` is `False`, so this form counts a `nan`; the naive
`(t > 1).sum()` counts neither a `nan` nor a negative magnitude. That naive
form is what `G01_WEAKEN` substitutes, and it is what goes RED.

`[MEASURED]`:

| reading | exceedances | range |
|---|---|---|
| `13 × 8` adversarial `(u, g)` sweep — endpoints exactly, one ulp either side, `u` to `±1e12`, `g` to `1e6` | **`0`** | `m ∈ [0.0, 1.0]` |
| the hop products `path_product(m)` induced by that sweep (clause 1's RHS) | **`0`** | — |
| BED-M's own `\|a\|` at `n=64, s=64` | **`0`** | min `0.0`, max `1.0`; `= 0` on `3968`, `= 1` on `128` |
| shipped `ArmSMPrime`, as-constructed, 8 seeds, random data | **`0`** each | `â_max = 1.000000`, `â_min = 0.000000`, all eight |

**Both closed endpoints are attained**, and not only in the sweep: BED-M's real
gates put `|a| = 0` on 3968 entries and `|a| = 1` on 128. `â_max = 1.000000` on
all eight untrained seeds is the **cap binding**, not a coincidence — the
magnitude head is a random linear layer whose output saturates both ends. That
is the reading `R1`'s three divergent seeds put at `285` with an open range.

### Planted negative — the rejected cap order, and `nan`

`ceq/arm_smprime.py::blend`'s docstring states the cap is applied **after** the
blend and that the other order lets `g > 1` extrapolate to a **negative**
magnitude. Built here as `lerp(1, clamp(u,0,1), g)` — cap **first** — over the
same sweep `[MEASURED]`:

| negative | exceedances | range |
|---|---|---|
| cap-first-blend-second | **`35`** | `[−999999.0, 3.0]` |
| a `nan` magnitude through `smp.magnitude` | **`1`** counted | — |

### Non-degeneracy of the PASS half

1. **The counter is counting something that could have been non-zero.** The
   identical sweep, **un-capped**, leaves `[0,1]` on **`61`** entries. Asserted,
   not stated.
2. **Both holes in the naive counter are exhibited.** Under `G01_WEAKEN` the
   one-sided form reads `0` on a set whose minimum is `−999999.0`, and `0` on a
   `nan` — both go RED. A `|a| ≤ 1` check that reports `0` exceedances on
   `−999999` is precisely the shape of control this repo has struck 14 times.

---

## 5. BIND D — the deterministic-mode assertion

**Method.** The flag is process-global and four other nodes share this tree, so
the whole probe runs in a **subprocess** with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
set **before** torch initialises. The subprocess is launched **twice**, so the
byte-equality claim is **cross-process** and not a cached tensor compared with
itself. `s = 64` — the arm's own scan length and the smallest shape that
instantiates the kernel. On the **device**, as stated.

Full probe output `[MEASURED]`, this box, this run:

```json
{
  "torch": "2.5.1+cu121",
  "flag_live": true,
  "cublas_workspace": ":4096:8",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "readout_ran": true,
  "within_process_bitwise": true,
  "sha": "e547261108d1fb83ff7dd5e95cf1785efb515927f1c4350fa8a1ce5ea925a053",
  "sha_one_ulp": "a783371e9dc963dbfba65bf63c97e5a34ce996b62283fc76d2652b0f2dd66735",
  "label_cell_residual": 3.552713678800501e-15,
  "label_cell_ran": true,
  "module_forward_nograd": "OK",
  "module_forward_bitwise": true,
  "module_forward_sha": "0f58a4f19d1519b59bc09904c68679149dfe9eb7f21bbab60a8c1690d0afdfa6",
  "module_fwd_bwd": "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. ...",
  "cumsum_f64":       "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, ...",
  "cumprod_f64": "OK",
  "cumprod_c128": "OK",
  "cumprod_backward": "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, ...",
  "path_product_f64": "OK",
  "path_product_c128": "OK",
  "hop_scan":         "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, ...",
  "warn_only_fwd_bwd": "OK",
  "warn_only_grad_bitwise": true,
  "warn_only_grad_finite": true,
  "warn_only_grad_sha": "c65a09aee7d82d3e3a9058bca3fde9a4a77f26a4453fc9da5c0ba9c2eae232ea"
}

second process sha              = e547261108d1fb83ff7dd5e95cf1785efb515927f1c4350fa8a1ce5ea925a053   [IDENTICAL]
second process module fwd sha   = 0f58a4f19d1519b59bc09904c68679149dfe9eb7f21bbab60a8c1690d0afdfa6   [IDENTICAL]
second process warn_only grad   = c65a09aee7d82d3e3a9058bca3fde9a4a77f26a4453fc9da5c0ba9c2eae232ea   [IDENTICAL]
```

**The exact exception text**, verbatim, in full, since the item asks for it:

```
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'. You can turn off
determinism just for this operation, or you can use the 'warn_only=True'
option, if that's acceptable for your application. You can also file an issue
at https://github.com/pytorch/pytorch/issues to help us prioritize adding
deterministic support for this operation.
```

**Per-callable verdict, re-measured on this tree and NOT inherited from
`V16_R1_DEVICE_READY.md`:**

| callable, cuda, `use_deterministic_algorithms(True)` | result |
|---|---|
| `torch.cumsum` float64 | **RAISES** |
| `torch.cumprod` float64 | **OK** |
| `torch.cumprod` complex128 | **OK** |
| **`torch.cumprod` BACKWARD** | **RAISES — the same kernel** |
| `smp.path_product` float64 / complex128 | **OK** |
| `smp.hop_scan` (the planted negative's route) | **RAISES** |
| the (L) read-out, `smp.label_cell` | **OK**, residual `3.552714e-15` on cuda |
| `ArmSMPrime.forward` under `no_grad` | **OK** |
| `ArmSMPrime` forward **+ backward** | **RAISES** |
| the same, with `warn_only=True` | **OK**, gradients finite and **bitwise reproducible in-process and across processes** |

**So the load-bearing fact, stated plainly: `cumprod` does NOT have the hole on
the forward and DOES have it on the backward.** `arm_smprime` is therefore
deterministic to **evaluate** on cuda and not deterministic to **train** there.
Every bind in G0.1 is a forward statement, so the item is green; a Kaggle
training run under strict determinism is not, and needs the `warn_only=True`
demotion, which was measured to work and to leave the gradient bitwise
reproducible across two processes.

### Non-degeneracy of the PASS half — both halves of it

1. **The regime is real.** An equality measured with the flag silently off
   proves nothing about determinism. The proof the flag is live is that
   `torch.cumsum` **raises in the same process, at the same moment**, with the
   text above — asserted, not assumed. `flag_live` is read back from
   `torch.are_deterministic_algorithms_enabled()` as a second witness.
2. **The comparator is not blind.** A sha equality that cannot fail is worth
   nothing. **One ulp** on a single drive (`torch.nextafter` on `b[1]`) moves
   the read-out's sha from `e547261108d1fb83…` to `a783371e9dc963db…` —
   measured, asserted. Under `G01_WEAKEN` the comparator is demoted to a
   3-decimal scalar summary, which reads `8.964` for **both** runs and goes RED
   on exactly this assertion.
3. **The backward hole is pinned by kernel name**, so a torch bump that closes
   it surfaces as a failing test rather than as nobody noticing.

---

## 6. FOUND BROKEN IN FILES THIS NODE DOES NOT OWN — filed, not fixed

**(1) `ceq/arm_smprime.py::label_cell` reads `1.44` on BED-M's real corpus with
HONEST settings.** `label_cell` scores **every** position against
`chain_label`, whose convention is `y_0 = 0`. The path-product read-out at
position `0` is `G_00 b_0 = b_0`. The two therefore agree at position `0` only
when `b_0 = 0`. `smp.bedm_draw` sets `b[0] = 0.0` for exactly that reason;
BED-M's own `make_equilibrium_batch` does **not** — it zeroes `b[s−1]`, the
query token's driver, not `b[0]`. Measured at `n=8, s=64` `[MEASURED]`:

| | value |
|---|---|
| `label_cell` residual, honest settings, real corpus | `1.440495e+00` |
| `max_i \|b_0\|` on the same draw | `1.440495e+00` — **the residual IS `b_0`** |
| the same with `b_0` zeroed | `2.965914e-16` |
| the bind as G0.1 states it (last position, bed's oracle) | `2.965914e-16` |

The bind is unaffected — G0.1 is read at the last position against the bed's
own oracle, which is where `equilibrium_oracle` is defined. **The hazard is for
any node that files a published cell through `label_cell` on the real corpus
and gets `1.44` where it expects `1e-16`.** Pinned by
`test_bind_a_label_cell_disagrees_with_the_bed_at_position_zero`, which asserts
the residual equals `max|b_0|` to `1e-12` and that zeroing `b_0` removes it, so
the mechanism is nailed and not merely described. **Not fixed:**
`ceq/arm_smprime.py` is not this node's file.

**(2) `V16_ARM_SMPRIME.md` (e) certifies the softmax corner with a statistic
that is vacuous for the `g` switch.** `Σ_j |W_ij| = 1.000000` at `β = 1` is an
identity for every gate; measured at `4.440892e-16` with `g = 1`. §3 has the
mechanism and the replacement. **Not fixed:** that file is another node's
record of another round.

**(3) `V16_R1_DEVICE_READY.md` §8 item 1 is half wrong in its training half.**
It proposes replacing `cumsum` with `cumprod` on the grounds that "`cumprod` is
deterministic on cuda". Measured here: true of the forward, **false of the
backward** — `cumprod`'s backward calls the identical `cumsum_cuda_kernel` and
raises. Any decision made from that sentence about a *training* run under strict
determinism is made from a false premise. **Not fixed:** not this node's file.

**(4) Not a defect, a scope note.** `ceq/beds/bed_1.py` is the multi-basin
committor bed; BED-M — the bed G0.1's label bind is stated on — lives in
`scale/negation_scope.py` (`make_equilibrium_batch` / `equilibrium_oracle`) and
has no module under `ceq/beds/`. This node read `bed_1.py` and measured against
`negation_scope.py`, which is where the bind is.

---

## 7. THE CALL

| item | verdict |
|---|---|
| oracle gates ⇒ label ≤ `1e-6` | **GREEN** — `5.919777e-16` / `7.550528e-16` / `7.550528e-16`, four planted negatives fire at `O(1)` or `nan`, sensitivity measured at gain `4.2773` |
| softmax corner row sums `1.000000` | **GREEN** — `2.220446e-16` over all 64 rows, `41/64` bitwise; fires on a `1e-6` move of `β` |
| `\|a\| ≤ 1`, zero exceedances | **GREEN** — `0` on every reading, both closed endpoints attained; counter fires at `35` on the rejected cap order and at `1` on a `nan` |
| deterministic-mode assertion | **GREEN on the forward path** — runs, bitwise identical across two processes. **The backward RAISES.** |

## **G0.1 — GREEN.**

All four binds hold on the tree at `ab5b48547884e04258276e6e808d5a71ea65f917`,
each with a planted negative that fires and each with a non-degeneracy check on
its PASS half, and the whole suite is shown capable of failing by the
`G01_WEAKEN` RED in §1.

**The one thing that must travel with this verdict to the Kaggle decision:**
G0.1's four binds are all forward-path statements and the deterministic-mode
PASS is a forward-path PASS. `arm_smprime` **cannot take a backward step on
cuda under `use_deterministic_algorithms(True)`** in torch `2.5.1+cu121` —
`cumprod`'s backward calls `cumsum_cuda_kernel`. A Kaggle **training** run in
that regime must either use `warn_only=True` (measured here: runs, gradients
finite, bitwise reproducible in-process and across two processes) or train on
cpu, and whichever is chosen is a decision that belongs in the run's header.

---

### LIMITS

Collected here, once, as the standing laws require.

- Bind D's cross-process byte equality is **two processes on one card in one
  session**. `V16_BAR_RECERT.md` flags that shape of claim as insufficient for
  a determinism guarantee, and it is cited here as evidence, not as a licence.
- The `warn_only=True` reading is **two backward passes in one process plus one
  cross-process sha**, at `n=4, s=16, d_model=8`. It says the demotion runs and
  reproduced at that shape; it is not a reproducibility certificate at training
  scale or training length.
- Bind B's `41/64` bitwise-exact row count is a property of this draw
  (`seed = 3`, `n = 64`, `d = 8`) and of the summation order; it is reported as
  a measured fact and is deliberately **not** a bar.
- Bind C's PASS is a property of `clamp` and carries no information on its own;
  the content is in the three negatives, and this is stated in §4 rather than
  left for a reader to notice.
- The perturbation gain `4.2773` is measured at `n=64, s=64, t*=2, seed=0` on
  the last hop only. It is a sensitivity for that draw, not a general
  conditioning constant.
- Bind A's `t* = 2` and `d = 24` are BED-M's `e3_t2` settings; no other rung of
  the ladder was measured and none is claimed.
- Nothing here was trained and no cell of R1′, R2 or any bed was produced,
  scored or filed (L-LEAN).
