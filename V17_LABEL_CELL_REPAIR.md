# V17 — G1.1: the `label_cell` BOS-slot repair, and the softmax corner's vacuous certificate

**Node G1.1.** Repairs the first defect `V17_G01_IDENTITY.md` §6 filed and was
forbidden to fix, and files the second.

---

## 0. LEAD CAVEAT

**Nothing here is a research reading.** No NRMSE, no cell of R1′/R2/any bed, no
seed pooling, no verdict, no training step, no GPU. Every number below is a
float64 CPU numeric taken on this box this run. The one code change is inside
`ceq/arm_smprime.py::label_cell`'s residual line; `ArmSMPrime.forward`, the path
product, `operator`, `readout`, `hop`, and every public signature are byte-for-byte
unchanged, so the wiring agent reading this module as a read-only dependency is
unaffected. **NO NEW CONSTRUCTIONS**: no arm, no oracle, no comparator was
invented — the repair *removes* a term from a `max`, and the tests call
`label_cell`, `readout`, `mutate`, `oracle_heads`, `operator` and
`make_equilibrium_batch` as shipped.

**One existing test breaks, and it is the one that pinned the defect.** Named in
§5. It is another node's file and was not touched.

**The repair does not make `chain_label` correct on every corpus.** At
`t_star=None` `chain_label` and BED-M's own `equilibrium_oracle` are genuinely
different labels, by `max|b_0|`; the repaired scorer reports that, which is §3.3
and is the reason candidate (b) is refused.

Tags: `[MEASURED]` this box this run · `[MODULE]` from a named function ·
`[INHERITED]` cited by file · `[ASSUMED]` with reason.

**Run identity.** `[MEASURED]`

| | |
|---|---|
| `git rev-parse HEAD`, start **and** end | `ab5b48547884e04258276e6e808d5a71ea65f917` |
| `git status --porcelain`, start | ` M ceq/hf/configuration_ceq.py`, ` M ceq/hf/modeling_ceq.py`, ` M ceq/hf/train.py`; untracked `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G08_G09_AUTOPILOT.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `tests/gate0/` |
| `git status --porcelain`, end | the same, **plus** ` M ceq/arm_smprime.py` and untracked `V17_LABEL_CELL_REPAIR.md` (this node's; `tests/gate0/` was already listed untracked and now contains `test_g11_label_cell.py`), **plus** untracked `V17_NOTEBOOK.md` and ` M ceq/hf/modeling_ceq.py`, which appeared mid-run and are another agent's — seven are working in this tree |
| torch / python / device | `2.5.1+cu121` / `3.11.9` / **cpu, float64 throughout** |
| no writing git command was run | correct |

---

## 1. THE TWO DEFECTS, IN ONE LINE EACH

| | defect | mechanism | status |
|---|---|---|---|
| **D1** | `ceq/arm_smprime.py::label_cell` read `1.440495` on BED-M's real corpus with honest settings | it scored the **BOS slot**, where the read-out is the *empty* path product and the residual is a constant function of the arm | **REPAIRED**, §3 |
| **D2** | `V16_ARM_SMPRIME.md` row (e) certifies the softmax corner with `Σ_j\|W_ij\| = 1.000000` at `β=1` | that quantity is `Z_i^{1−β}`: the gate cancels against its own normalizer, so it sees `β` and **cannot see `g`** | **FILED**, §4 (documentation defect; no code change) |

---

## 2. D1 — THE RED

`tests/gate0/test_g11_label_cell.py`, run against `label_cell` **before** the
edit. 6 failed, 7 passed. `[MEASURED]`

```
$ python -m pytest tests/gate0/test_g11_label_cell.py -q -s

    def test_the_real_corpus_residual_reads_the_hops_and_not_max_abs_b_zero():
        ...
        rec = smp.label_cell(a, b, mutation="none")
>       assert rec["residual"] <= LABEL_BAR, rec["residual"]
E       AssertionError: 1.4404945373535156
E       assert 1.4404945373535156 <= 1e-06

    def test_the_planted_negatives_still_fire_after_the_repair(mutation):
        ...
>       assert honest <= LABEL_BAR, honest
E       AssertionError: 1.4404945373535156          # all four mutations

    def test_the_repair_is_inert_where_the_bos_convention_already_held():
>             f"   residual_bos = {rec['residual_bos']:.6e}")
E           KeyError: 'residual_bos'

FAILED ...::test_the_real_corpus_residual_reads_the_hops_and_not_max_abs_b_zero
FAILED ...::test_the_planted_negatives_still_fire_after_the_repair[drop_phase]
FAILED ...::test_the_planted_negatives_still_fire_after_the_repair[drop_magnitude]
FAILED ...::test_the_planted_negatives_still_fire_after_the_repair[beta_one]
FAILED ...::test_the_planted_negatives_still_fire_after_the_repair[exp_scan]
FAILED ...::test_the_repair_is_inert_where_the_bos_convention_already_held
6 failed, 7 passed in 0.89s
```

The same run, **after** the edit: `13 passed in 0.51s`. `[MEASURED]`

Note what the four negative rows say on their own: with the defect present, the
**honest** cell read `1.440495`, so BIND 1's rejection region on the real corpus
was being read against a failing positive. `V17_G01_IDENTITY.md` §"instrument
changed" records having hit exactly this and having routed around it by reading
the last position against the bed's own oracle.

---

## 3. D1 — THE REPAIR

### 3.1 The mechanism, established from the two definitions

`ceq/arm_smprime.py`'s module docstring defines the (L) corner read-out as
`O_i = Σ_j G_ij V_j` with `V = b` unrescaled and
`G_ij = prod_{k=j+1}^{i} a_k`. `ceq/arm_phase.py::chain_label` defines
`y_0 = 0`, `y_i = a_i y_{i−1} + b_i`, and its loop is `for i in range(1, s)` —
so slot 0 is the chain's **initial state**, which carries no drive. Expanding:

```
y_i = Σ_{j=1..i} G_ij b_j          (chain_label: the sum STARTS AT 1)
O_i = Σ_{j=0..i} G_ij b_j          (the arm: the sum STARTS AT 0)
O_i − y_i = G_i0 · b_0             exactly, at every row i
```

At `i = 0` this is `G_00 b_0 = b_0`, because `G_00` is the **empty product**.
That term contains no gate, no `m`, no `theta`, no route and no `β`.

**Measured consequence, real corpus `n=8, s=64, d=24, t_star=2, seed=0`:**
`[MEASURED]`

| slot-0 read-out vs `b_0`, by setting | `max\|O_0 − b_0\|` |
|---|---|
| `none` | `0.000e+00` |
| `drop_phase` | `0.000e+00` |
| `drop_magnitude` | `0.000e+00` |
| `beta_one` | `0.000e+00` |
| `exp_scan` | `nan` — from the BOS's own `log m_0 = −inf` (`max\|a_0\| = 0.0` on this corpus), a quantity `path_product` never reads |

**No setting of the arm can move the slot-0 term.** A residual term that is
invariant under every mutilation of the construction is not a reading of the
arm; it is a reading of what the caller put in slot 0. That is the argument, and
it is the same shape as the vacuity the round has struck fourteen times — here
sitting inside a *positive* rather than a control.

### 3.2 The change

`ceq/arm_smprime.py::label_cell`, one line replaced (comment elided):

```python
-    record["residual"] = float((out - tgt).abs().max())
+    record["residual_bos"] = float((out[..., BOS] - tgt[..., BOS]).abs().max())
+    record["residual"] = float((out[..., BOS + 1:] - tgt[..., BOS + 1:]).abs().max())
```

`BOS = 0` is the module's own constant. **The excluded term is reported, not
deleted** — `residual_bos` is a new record key, and §3.3 is why that matters.

**Nothing else moves.** `residual_bos` is outside `SMP_FIELDS`; `label_cell` is
not among the `callables` `cell_manifest` fingerprints
(`operator, readout, hop, path_product, oracle_heads`) and `residual` is not in
`identity_manifest.CONFIG_FIELDS`. So **no published manifest hash moves under
this repair**, pinned by
`test_the_repair_does_not_move_a_published_manifest_hash`, which also re-asserts
that the hash still separates all five settings.

### 3.3 Why candidate (b) — "zero `b_0` in the draw" — is wrong

Three reasons, the third measured.

1. **It changes the data to fit the instrument.** The repo's standing law is
   that a scorer is repaired, never the corpus. `label_cell` does not own the
   draw; it receives `(a, b)`. Zeroing `b_0` inside it would mutate a caller's
   tensor and publish `v_max` for a corpus the caller never passed.
2. **The arm is the party that is *right* at slot 0.** BED-M's own
   `scale/negation_scope.py::equilibrium_oracle` scans `for i in range(s)` from
   `z = 0`, so **it absorbs `b_0`** — `z_0 = b_0`, which is exactly the arm's
   `O_0`. It is `chain_label` that carries the drive-free-slot-0 convention.
   Zeroing `b_0` would therefore delete a drive the corpus's own label reads.
3. **It would hide a real label disagreement.** `[MEASURED]`, at
   `t_star=None` (full-length chain, `head = 0`, so the gate at position 1 is
   live), `n=8, s=64, seed=0`:

   | quantity | reading |
   |---|---|
   | `max\|G_i0\|` over `i ≥ 1` | `1.000000` — **column 0 is live**; `b_0` reaches every later row |
   | rows `1..` residual, repaired scorer | `1.440495e+00` — **reported** |
   | `\|chain_label[−1] − equilibrium_oracle\|` | `1.440495e+00` — the two are **genuinely different labels** here |
   | `\|read-out[−1] − equilibrium_oracle\|` | `3.500426e-14` — **the arm computes the bed's label** |
   | rows `1..` with `b_0` zeroed | `2.915361e-14` — **a PASS** |

   Under (b) the scorer would read `2.9e-14` and the disagreement would vanish
   from the record. Under the repair it reads `1.440495` and a reader is sent to
   look. A repair that makes a real disagreement invisible is not a repair.

`[ASSUMED]` — that `t_star=None` is a setting a caller may reach: it is
`make_equilibrium_batch`'s own documented default (`"t_star=None means the full
length"`), so no caller has to be unusual to reach it.

### 3.4 Before / after residuals, both draws

`[MEASURED]`, cpu, float64.

| draw | `b_0` | residual **before** | residual **after** | `residual_bos` after |
|---|---|---|---|---|
| **toy** `bedm_draw(seed=15, s=64)` | `0.0` | `3.552714e-15` | `3.552714e-15` — **unmoved** | `0.000000e+00` |
| **toy** `band_draw(seed=15, s=64)` | `0.0` | `3.552714e-15` | `3.552714e-15` — **unmoved** | `0.000000e+00` |
| **real corpus** `make_equilibrium_batch(8, 64, 24, t_star=2, seed=0)` | `max\|b_0\| = 1.4404945373535156` | **`1.4404945373535156`** | **`2.965913655728132e-16`** | `1.440495e+00` |

The defect was invisible on the toy draw for the reason `bedm_draw`'s own
docstring gives — it sets `b[0] = 0.0` — and the repair is inert there. On the
real corpus the residual falls **by 15.7 orders of magnitude**, and
`2.965913655728132e-16` is bit-identical to the `b_0`-zeroed reading
`V17_G01_IDENTITY.md` §6 published, which is the independent check that nothing
but the BOS term was ever wrong.

### 3.5 Non-degeneracy on every PASS half

A residual that dropped a position could have dropped the bind's teeth with it.
Three checks, all on the **real corpus**, all `[MEASURED]`:

**(i) Planted perturbation of the arm, corpus label held fixed** — gate at the
last position moved by `eps`:

| `eps` | repaired residual |
|---|---|
| `1e-12` | `2.422063e-12` — inside the `1e-6` bar |
| `1e-9` | `2.421852e-09` — inside |
| `1e-6` | `2.421852e-06` — **outside** |
| `1e-3` | `2.421852e-03` — **outside**, and monotone in `eps` |

**(ii) All four shipped planted negatives still fire**, now against an honest
positive of `2.965914e-16` instead of `1.440495`:

| mutation | residual |
|---|---|
| `drop_phase` | `4.843705e+00` |
| `drop_magnitude` | `2.051604e+01` |
| `beta_one` | `1.614568e+00` |
| `exp_scan` | `nan` |

**(iii) The manifest still separates all five settings** — five distinct hashes,
and `residual_bos` is absent from `smp_values`, so the identity is unmoved.

---

## 4. D2 — THE VACUOUS CORNER CERTIFICATE, FILED

### 4.1 Which quantity the statistic cannot see, and why the identity holds

`ceq/arm_smprime.py::operator` computes `W_ij = num_ij / Z_i^β` with
`num_ij = G_ij e_ij` and `Z_i = Σ_j R_ij e_ij`, where `R = prod m` is the
modulus row and `e_ij > 0` is the real content term. Clause 1 of `#2` re-stated
is `|G_ij| = R_ij`. Therefore

```
Σ_j |W_ij|  =  Σ_j R_ij e_ij / Z_i^β  =  Z_i / Z_i^β  =  Z_i^{1−β}   ≡ 1 at β = 1
```

**The gate cancels against its own normalizer.** `m`, `theta` and `g` enter the
numerator and `Z` in exactly the same way and divide out. The statistic is a
function of `β` alone — it is `1.000000` at `β = 1` for **every** gate
whatsoever, so it certifies the `β` half of the corner and is **structurally
incapable** of certifying the `g ≡ 0` half. It is not a weak test of `g`; it is
not a test of `g`.

`V16_ARM_SMPRIME.md` §4.5 does carry a non-vacuous `g` negative elsewhere
(`max|gap| > 0.1` against `ceq/lm.py` with the gate left on), so the document is
not uniformly blind. The defect is in **row (e)**, the summary line a reader
reads as "the switches are certified", and in §5.1's table, both of which read
the corner through `Σ_j|W_ij|`.

### 4.2 Both statistics on the same draw

Draw: `tests/gate0/test_g01_identity.py::_corner_inputs` — `n=64`, `d=8`,
`seed=3`, float64, cpu. `[MEASURED]`

| `g` at `β = 1` | `max\|Σ_j\|W_ij\| − 1\|` **(row (e))** | `max\|Σ_j Re W_ij − 1\|` **(replacement)** |
|---|---|---|
| `0.00` — **the corner** | `2.220446e-16` | `2.220446e-16` |
| `0.25` | `4.440892e-16` | **`1.212002e+00`** |
| `0.50` | `4.440892e-16` | **`1.481515e+00`** |
| `0.75` | `3.330669e-16` | **`1.346664e+00`** |
| **`1.00` — the gate fully on** | **`4.440892e-16`** | **`1.750255e+00`** |
| `2.00` | `2.220446e-16` | **`1.445854e+00`** |

Row (e)'s statistic never leaves `4.5e-16` across the whole sweep. This
reproduces `V17_G01_IDENTITY.md`'s `4.440892e-16` at `g = 1` and its
`1.750255` for the real row sum, on the same draw, to every printed digit.

**Blind to the gate itself, not only to `g`** — `β = 1`, `g = 1`, three
unrelated gates: `[MEASURED]`

| gate | `Σ\|W\| − 1` | `Σ Re W − 1` |
|---|---|---|
| `m ~ U(0,1)`, `θ ~ U(−π,π)` | `2.220446e-16` | `1.507157e+00` |
| `m = 1, θ = π` — **BED-M's own `−1`** | `2.220446e-16` | `1.502297e+00` |
| `m = 0` — fully annihilating | `0.000000e+00` | `0.000000e+00` |

### 4.3 The replacement: `Σ_j Re W_ij`

`Σ_j Re W_ij` — the row sum **over the real part**, which is what a softmax row
sum means. It is not a new construction: it is the same reduction taken on
`W.real` instead of `W.abs()`, and `tests/gate0/test_g01_identity.py`'s
`test_bind_b_the_row_sum_fires_off_the_g_corner_only_on_the_real_part` already
reads it.

**It is a strict strengthening, not a swap.** At `g = 0` the operator is real
and non-negative — `max|Im W| = 0.0` exactly — and `[MEASURED]`
`torch.equal(W.real, W.abs())` is **True entrywise**, so the replacement reads
exactly what row (e) reads at the corner, and keeps all of its `β` sensitivity:

| `β` at `g = 0` | `Σ\|W\| − 1` | `Σ Re W − 1` |
|---|---|---|
| `0.0` | `1.377300e+02` | `1.377300e+02` |
| `0.5` | `1.077837e+01` | `1.077837e+01` |
| `0.999999` | `4.932542e-06` | `4.932542e-06` — still outside the `5e-7` bar |
| `1.0` | `2.220446e-16` | `2.220446e-16` — **PASS** |

The two row sums differ only by `2.220446e-16`, which is float64 reduction order
over a strided `.real` view, not a difference in the quantity.

**RED for D2, reproducible.** `tests/gate0/test_g11_label_cell.py` carries a
`G11_ROW_E=1` switch, in the house pattern of `test_g01_identity.py`'s
`G01_WEAKEN`: under it the corner is certified with row (e)'s statistic and the
planted `g` walks straight through. `[MEASURED]`

```
$ G11_ROW_E=1 python -m pytest tests/gate0/test_g11_label_cell.py::\
test_row_e_is_an_identity_in_beta_and_cannot_see_the_g_switch -q

E       AssertionError: the corner statistic did not separate g=0.25: 4.440892098500626e-16
E       assert not 4.440892098500626e-16 <= 5e-07
1 failed in 0.67s
```

Unset, the same test is green: the identity is asserted **as an identity** over
the whole `g` sweep, and the replacement separates every `g ≠ 0`.

**The replacement's own limit, on the record rather than discovered later.** On
the fully annihilating gate (`m = 0`) every row is a delta, so it genuinely sums
to `1` in *both* statistics — `0.000000e+00` above. `Σ Re W` is therefore a
statistic that **moves when `g` moves**, which is the job D2 names; it is not by
itself a complete certificate of the `g ≡ 0` corner. `test_g01_identity.py`'s
own corner test already pairs it with a direct assertion that `g = 0` gives
`m ≡ 1` and `θ_eff ≡ 0`, which closes that gap.

---

## 5. WHAT BROKE, AND WHAT A READER MUST RE-READ

### 5.1 One existing test fails, and it is the one that pinned D1

**`tests/gate0/test_g01_identity.py::test_bind_a_label_cell_disagrees_with_the_bed_at_position_zero`**,
line 209: `[MEASURED]`

```
>       assert abs(cell - b0) <= 1e-12, (cell, b0)      # the residual IS b_0
E       AssertionError: (2.965913655728132e-16, 1.4404945373535156)
E       assert 1.4404945373535154 <= 1e-12
```

**No assertion was weakened and that file was not touched.** The test's own
docstring says *"Filed rather than fixed: `ceq/arm_smprime.py` is not this
node's file"* — it asserts the defect is present, so repairing the defect must
fail it. Its owner should re-point it: `cell` is now `2.965913655728132e-16` and
the `max|b_0|` claim moves to the new `residual_bos` key, i.e.
`assert abs(rec["residual_bos"] - b0) <= 1e-12`. The other two assertions in
that test (`cell_zeroed <= 1e-6`, `last <= LABEL_BAR`) still hold unchanged.

### 5.2 Suite counts, before and after

Both at `HEAD = ab5b485…`, `git status --porcelain` as in §0. `[MEASURED]`

| suite | before | after |
|---|---|---|
| `tests/arm_smprime` | **52 passed**, 0 failed (8.78s) | **52 passed**, 0 failed (12.26s) |
| `tests/gate0` | **140 passed, 4 failed** (68.46s) | **152 passed, 5 failed** (73.99s) |

The `tests/gate0` delta decomposes exactly: `+13` from the new
`test_g11_label_cell.py` (all passing), `+1` failure from §5.1. The **4
pre-existing failures are all `tests/gate0/test_g06_kcert.py`**
(`KeyError: 'zero_step'`, `KeyError: 'determinism'`, and two siblings), they
read `results/k_cert_local.json` which is an untracked artifact another agent is
rewriting, they failed identically before this change, and they are unrelated to
it.

### 5.3 Published cells a reader must re-read

**None of the published R1/R1′ bind rows.** `scripts/v15_r1.py:481` calls
`label_cell` only on `arm_smprime.bedm_draw` / `arm_pl.draw`, both of which set
`b[0] = 0`, so `residual_bos = 0.0` and the headline number is byte-identical
before and after (§3.4). The same holds for the `label_cell_residual` field in
`test_g01_identity.py`'s cuda determinism probe (`bedm_draw`), for
`V16_ARM_SMPRIME.md` row (a) (whose residuals come from the bed via
`readout` at the last position, not from `label_cell`), and for rows (f) and (h)
(band/`bedm` draws).

**Three lines in `V17_G01_IDENTITY.md` now describe a repaired scorer** — that
document's own §6 filing, which is correct as a filing and stale as a present
reading:

| site | text | status |
|---|---|---|
| `V17_G01_IDENTITY.md:142` | *"The honest cell reads `1.44` on BED-M's real corpus"* | now `2.965914e-16` |
| `V17_G01_IDENTITY.md:158` | `label_cell on the real corpus = 1.440495e+00` | now `2.965914e-16`, with `residual_bos = 1.440495e+00` |
| `V17_G01_IDENTITY.md:491-492` | *"the residual IS `b_0`"* | now the `residual_bos` column |

`V16_ARM_SMPRIME.md` row (e) and §5.1's `β` table are **not withdrawn** — every
number in them is correct — but they must be read as certifying `β` only, per
§4. That file is not this node's to edit.

### 5.4 The same mechanism is latent in two sibling scorers

`[MEASURED]` by inspection, not repaired — neither file is this node's:

- `ceq/arm_phase.py:428` — `record["residual"] = float((out - y).abs().max())`
- `ceq/arm_pl.py:267` — `record["residual"] = float((out - y).abs().max())`

Both score every position against `chain_label` exactly as `arm_smprime` did. It
is **latent, not live**: `arm_phase.draw`, `arm_phase.band_draw` and
`arm_pl.draw` all set `b[0] = 0`, so no shipped call site reaches the defect.
The first caller that hands either of them a corpus whose slot 0 carries a drive
gets `max|b_0|` back, silently. Filed here for those files' owners.

---

## 6. THE TWO LEDGER LINES

`MISTAKES.md` is another node's file and was not edited. Paste-ready, one line
each, naming the mechanism; the numbers `V-26` / `M-22` are the next free ones
after `V-25` and `M-21` and are the ledger owner's to confirm.

```
### V-26. A residual term the arm cannot move, scored as if it measured the arm
### M-22. A corner certificate read through a quantity in which the switch it certifies cancels
```

---

## 7. FILES

| file | change |
|---|---|
| `ceq/arm_smprime.py` | one residual line replaced by two, plus its comment. No signature, no `forward`, no path product touched. |
| `tests/gate0/test_g11_label_cell.py` | new, 13 tests. RED before the repair on 6, green after on all 13; `G11_ROW_E=1` is D2's reproducible RED. |
| `V17_LABEL_CELL_REPAIR.md` | this file. |

Nothing else in the tree was written.
