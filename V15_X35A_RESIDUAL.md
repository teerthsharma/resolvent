# V15 X₃₅a — THE RESIDUAL INSTRUMENT

Node SATURN-3, CEQ v15.1 composition round, amended mid-node by
`CEQ_V15_2_DELTA.md`. Spec: `CEQ_V15_1_DELTA.md` §X₃₅a THE INSTRUMENT and
§KILLS.

Code: `ceq/x35/residual.py`, `ceq/x35/__init__.py`, the plant field in
`ceq/beds/bed_k.py`. Tests: `tests/x35/test_residual_onset.py`.
Provenance for every number below: parent commit `caec1ed`, Python 3.11.9,
numpy 1.26.4, Windows-10-10.0.26200, float64, single machine.

---

## VERDICT

**X₃₅ is NOT VOID at this node's scope, and the scope is narrow.**

The no-plant residual is flat — measured, with a rejection region demonstrated
on the same code path — so onsets called by this instrument are readable. The
bar was set against an **exact visible model** (the oracle), which is the only
configuration in which "flat under no plant" is checkable exactly and is the
order `CEQ_V15_1_DELTA.md`'s own CONSEQUENCE FOR THE ROUND'S ORDER asks for.
It is **not** a property that transfers: a trained arm's `z_model` carries
approximation error the oracle does not, and the flatness reading below must be
**re-measured** before any onset read off an arm's residual means anything. The
control at §5 shows what that failure looks like when it happens.

---

## 1. THE INSTRUMENT

`r = z_obs − z_model(visible)`; onset is the **first index whose residual
energy `r_i²` exceeds a threshold**, each index compared independently, nothing
accumulated across indices (`ceq/x35/residual.py::detect`). That is the
Shewhart chart the delta specifies. **No CUSUM path exists in the module**, and
no CUSUM number is reported: the delta records CUSUM's dependence inflation as
having killed X₂₆ with the memoryless detector surviving at matched `ARL₀`,
and a second detector sitting in the file invites exactly the isolated
comparison that result rules out. The `ARL₀` axis itself is reported (§4) so a
matched comparison remains possible for whoever needs one.

Output per run, `detect` → `(onset, magnitude, gate_estimates, residual,
energy, threshold)`:

- `onset` — first exceeding index, or `None`.
- `magnitude` — RMS residual from the onset on. Biased **up** by the noise
  variance inside the root (it estimates `sqrt(m² + sd²)`); the bias is
  reported rather than subtracted, because subtracting an assumed noise
  variance puts a second unverified model inside the instrument.
- `gate_estimates` — the downstream per-index drive the posited latent node
  must carry for X₃₅b's re-propagation: the residual from the onset on, exactly
  0 upstream of it.
- `residual`, `energy` — **the whole field**, returned deliberately (see §7).

### What `z_model(visible)` is

`ceq/x35/residual.py::oracle_visible` = `bed_k.rebuild(bed, bed["b"])` — the
generator's own analytic forward pass over the **visible** variables `b`/`pos`.
It never reads `bed["plant"]`. Nothing is trained anywhere in this node;
threshold calibration is the only fitting and is the instrument's own.

The exactness has a consequence sharp enough to state as a test
(`test_oracle_residual_does_not_depend_on_the_kernel`): with an exact visible
model,

    r  =  observation noise  +  planted latent

**identically**, so the residual is the same on the delay bed and the power-law
bed at the same seed. The bar set here is therefore a statement about the
*observation model*, not about either kernel — which is the precise reason it
does not survive the substitution of a trained arm, whose error is
kernel-dependent and is not zero.

---

## 2. RED

Test file authored and run first, before `ceq/x35/residual.py` existed:

```
$ python -m pytest tests/x35 -q
=================================== ERRORS ====================================
______________ ERROR collecting tests/x35/test_residual_onset.py ______________
ImportError while importing test module 'C:\Users\seal\Desktop\New folder (32)\tests\x35\test_residual_onset.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\x35\test_residual_onset.py:60: in <module>
    from ceq.x35 import residual as rx
E   ImportError: cannot import name 'residual' from 'ceq.x35' (unknown location)
=========================== short test summary info ===========================
ERROR tests/x35/test_residual_onset.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.49s
```

### SECOND RED — the first implementation failed two of its own tests

Excerpted from `python -m pytest tests/x35 -q -s`; the two assertion lines and
the summary are verbatim, the tracebacks between them are elided.

```
E   AssertionError: a visible model truncated at lag 64 still passes the
    no-plant false-alarm gate (FAR=0.26666666666666666): the gate has no
    rejection region and every onset it licenses is uninterpretable
E   assert 0.26666666666666666 > 0.5
...
>       assert np.allclose(p["z"] - bed_k.rebuild(p, p["b"]), u, rtol=0, atol=0)
E       AssertionError: assert False
FAILED tests/x35/test_residual_onset.py::test_flatness_gate_has_a_rejection_region[64]
FAILED tests/x35/test_residual_onset.py::test_manifest_records_plant_ground_truth_on_both_kinds
2 failed, 12 passed in 3.29s
```

Both are recorded rather than quietly repaired, because one of them is a bar
that moved:

1. **The exactness claim was wrong, not the code.** `(K@b + u) − K@b` is not
   `u` bitwise for nonzero `u`; it is `u` to float64 rounding. It *is* exact
   where `u` is exactly `0.0`, since adding `+0.0` returns the operand bitwise
   — and that is why the `sd = 0` onset is exact rather than approximately
   exact. The test now asserts `atol=1e-12` on the whole field and **bitwise
   zero on the pre-plant prefix**, which is the claim that carries the
   must-fire.

2. **The rejection-region bar was a guess and it was amended.** `FAR > 0.5`
   had no derivation behind it. It is now `FAR > 10·α`, derived from the rate
   the gate claims to hold under a good visible model rather than from the
   realised 0.2667. The amendment is written into the test's own docstring, and
   both realised rates are printed beside it, because moving 0.5 → 0.1 after
   seeing 0.2667 is the shape of `MISTAKES.md` M-2 if it is done silently. What
   makes it not M-2: the new bar is a function of `α` alone, and 0.2667 played
   no part in its value.

---

## 3. GREEN

```
$ python -m pytest tests/x35 -q -s

[MUST-FIRE 1] true=36 onset(sd=0)=36 onset(sd=0.05)=36 mag_hat=0.9809 (m_true=1.0) tau(sd=0)=0 tau(sd=0.05)=0.0382564
[MUST-FIRE 1 / 500 seeds @ sd=0.05] within+/-1=0.9900 (bar 0.95, predicted ~0.976) offset 95% CI=[0, 1] offsets={0: 421, 1: 74, 2: 5} misses=0
[SUPERPOSITION / 200 seeds] sources at 36 and 80; first-onset within+/-1 of 36: 0.9900
[MUST-FIRE 2] measured FAR=0.0120 (24/2000) vs calibrated alpha=0.01 | per-index rate=9.375e-05 ARL0=1.067e+04 indices
[MUST-FIRE 2 @ sd=0] tau=0.0 max|r|=0.0 FAR=0.0 (0/200)
[FLATNESS / oracle] energy-vs-index corr=-0.1215 half_ratio=0.9747 mean_energy=0.00249003 (noise variance 0.0025)
[FLATNESS / truncated L=8] FAR=1.0000 corr=+0.9063 half_ratio=1.673 mean_energy=0.0298
[FLATNESS / truncated L=64] FAR=0.2667 corr=+0.8988 half_ratio=2.175 mean_energy=0.003975
[ZERO-MAGNITUDE PLANT] detection rate=0.0100 (5/500) vs alpha=0.01
[M-2] in-sample FAR=0.0095 (n=2000) fresh-draw FAR=0.0100 (n=500) alpha=0.01

14 passed in 3.31s
```

`python -m pytest tests/beds -q` → **15 passed in 0.99s**, unchanged from before
the plant field was added.

---

## 4. MUST-FIRE 1 — planted hidden cause ⇒ onset within ±1

Bed: `delay`, `d = 5`, `n = 128`. Plant: one latent source switching on at
**true index 36** with magnitude `m = 1.0`. Evaluation seed block
`[1 000 000, …)`, disjoint from the calibration block `[0, 2000)`.

| noise `sd` | threshold `τ` | onset (seed 1 000 000) | true index | offset |
|---|---|---|---|---|
| 0.00 | `0.0` | **36** | 36 | **0** |
| 0.05 | `0.0382564` | **36** | 36 | **0** |

`magnitude` at `sd = 0.05` reads **0.9809** against `m_true = 1.0`.

The delta's reference is `[RUN: exact at 0 noise, 36 vs 37 at sd 0.05]`. The
`sd = 0` half reproduces exactly and is **deterministic, not lucky**: with an
exact visible model the pre-plant residual is bitwise `0.0`, the calibrated
threshold is `0.0`, and the strict `>` comparator fires at the first index the
latent touches, for every seed. The `sd = 0.05` half is a distribution, so one
seed is one draw (`MISTAKES.md` M-4). Over **500 planted seeds**:

```
offsets  {0: 421, 1: 74, 2: 5}   misses: 0
within ±1 = 0.9900   (pre-registered bar 0.95, analytic prediction ~0.976)
measured 95% interval of onset offsets = [0, 1]
```

Named instances, so the delta's off-by-one shape is checkable rather than
asserted — reported by rule, not selected: **seed 1 000 012** is the first seed
in the evaluation block whose `sd = 0.05` onset is **37** against a true 36,
which is the delta's `36 vs 37` reproduced. **Seed 1 000 006** is the first
seed in the block at which the noisy onset disagrees with the noiseless one at
all, and it disagrees by **+2** (onset 38) — one of the 5 tail draws above,
listed because reporting only the +1 instance would hide the tail that the
±1 claim is 99.00% and not 100%.

**The bar's provenance.** 0.95 was fixed from the analytic prediction before
the run: at `sd = 0.05` the threshold is the 0.99 quantile of the run-max of
`(0.05·g)²` over 128 iid normals, `(2Φ(x)−1)^128 = 0.99` → `x ≈ 3.94` →
`τ ≈ 0.0388`; the detector misses index `t*` only when `|u| < √τ ≈ 0.197`,
probability ≈ 0.156, and misses `t*` and `t*+1` with probability ≈ 0.024, i.e.
within ±1 ≈ 0.976. The realised `τ` is **0.0382564** against the predicted
0.0388, and the realised offset-0 fraction is **0.842** against the predicted
0.844. The bar was not recomputed from the realised 0.9900.

### Superposition (`CEQ_V15_2_DELTA.md` must-fire 2)

Two planted sources, at 36 and 80, `m = 1.0` each, summed into `z`. Over 200
seeds the first-onset lands within ±1 of **36** at **0.9900**.

This is a **limit of the onset detector, stated rather than left for the wave
node to discover**: a Shewhart onset can only ever call the *earliest* source,
because the second switches on inside a residual that is already above
threshold. Separating superposed sources is what `CEQ_V15_2_DELTA.md`'s exact
solve `(I − A)r` and time-reversal `Wᵀr` are for. The manifest carries
`plant["indices"] == [36, 80]` and `plant["magnitudes"]` so that node scores
against the right ground truth, with `plant["index"]` naming the earliest —
the only one this instrument answers about.

---

## 5. MUST-FIRE 2 — NO plant ⇒ no onset called. THE SCOPE CONTROL.

Threshold calibrated on **2000 no-plant runs**, seeds `[0, 2000)`, as the
`1 − α` quantile of run-max residual energy at `α = 0.01`. Scored on **2000
fresh no-plant runs**, seeds `[1 000 000, 1 002 000)` — disjoint by
construction, and asserted disjoint in
`test_threshold_is_not_scored_on_the_block_it_was_calibrated_on`.

| quantity | value |
|---|---|
| calibrated target `α` | **0.01** |
| **measured false-alarm rate, fresh draw** | **0.0120 (24 / 2000 runs)** |
| in-sample rate on the calibration block | 0.0095 (19 / 2000) |
| per-index exceedance rate | `9.375e-05` |
| `ARL₀` | `1.067e+04` indices |

24 alarms against an expected 20 is `+0.9` binomial sd
(`sd = √(α(1−α)/2000) = 0.0022`, i.e. 4.5 runs). The in-sample rate is printed
beside the fresh one rather than substituted for it, and is `≤ α` **by the
quantile's own construction** — which is exactly why it is not evidence and the
fresh block is.

At `sd = 0`: `τ = 0.0`, `max|r| = 0.0` bitwise across the field, **0 alarms in
200 runs**.

### The residual is flat

Pooled over 200 fresh no-plant runs at `sd = 0.05`:

| statistic | measured | flat means | null sd |
|---|---|---|---|
| energy-vs-index Pearson `corr` | **−0.1215** | 0 | 0.089 (`1/√(n−1)`, `n = 128`) |
| second-half / first-half energy ratio | **0.9747** | 1 | 0.0177 |
| mean residual energy | **0.00249003** | 0.0025 (`sd²`) | — |

`corr` is 1.4 null sd from zero, the half-ratio 1.4 sd from one, and the mean
energy matches the observation-noise variance to three figures. The no-plant
residual is flat.

### The flatness gate has a rejection region — `MISTAKES.md` V-10

Under an exact visible model the residual is flat **by construction**, so the
flatness reading above is worth nothing unless the same gate, on the same code
path, is shown to reject a model that deserves rejecting. The control is
`truncated_visible(L)`: the same power-law kernel (`H = 0.75`) with every lag
beyond `L` dropped — the error a finite-context arm actually makes — with **no
plant anywhere** and the threshold calibrated on the untruncated model.

| visible model | FAR (no plant) | vs `α` | `corr` | half-ratio | mean energy |
|---|---|---|---|---|---|
| oracle (exact) | 0.0120 | 1.2× | −0.1215 | 0.9747 | 0.00249 |
| truncated `L = 64` | **0.2667** | **27×** | +0.8988 | 2.175 | 0.003975 |
| truncated `L = 8` | **1.0000** | **100×** | +0.9063 | 1.673 | 0.0298 |

The rejection is **graded in the model's error**, which is the useful part: an
arm whose residual sits between these rows gets a number, not a verdict. `L = 8`
drops most of the long-memory tail and alarms on every single run; `L = 64`
drops only the far tail and still alarms at 27× the calibrated rate. This is
`CEQ_V15_1_DELTA.md`'s VOID condition firing on demand.

### A plant of magnitude zero is not detected

The manifest says `present=True, index=36` while `magnitude=0.0` makes `z`
**bitwise identical** to the unplanted label at the same seed
(`test_zero_magnitude_plant_leaves_z_bitwise_unchanged`). Detection rate over
500 such runs: **0.0100 (5/500)** — the false-alarm rate, not `~1`. This
separates "detects a plant" from "detects the plant field being set".

The mirror of the same discipline, which `CEQ_V15_2_DELTA.md` raises against
the author's own KK probe (`0.000` on a planted anticipating kernel ⇒ VOID, a
test that cannot fire measures nothing): a detector that fires on planted data
*and* on unplanted data has the same defect pointing the other way. The
0.0120-against-0.0100 pair above, with its 2000-run count, is what excludes it.

---

## 6. THE KILL — explicit call

> *"no-plant residual fails flatness ⇒ X₃₅ VOID until the visible model is good
> enough that its residual means something (the detector cannot outrun the
> model it subtracts)."*

**NOT VOID**, on the measured flatness of §5, **at the oracle's scope only.**

Three conditions attach, and they are conditions, not hedges:

1. **The bar was set against an exact visible model.** The oracle is the
   generator's own forward computation, so the no-plant residual is exactly the
   observation noise and flatness is exact rather than estimated. That is the
   right order — the instrument's bar is fixed before any arm's residual is read
   — and it is also the reason the result is narrow.
2. **A trained arm will have approximation error the oracle does not.** The
   no-plant flatness must be **re-measured** on the arm's own residual before
   any onset read off it is admissible. The §5 control gives the re-measurement
   its calibration: FAR at 27× `α` was already a clear rejection.
3. **The oracle residual does not depend on the kernel** (§1), so this reading
   generalises across BED-K's two kinds and generalises to *no* arm.

`CEQ_V15_2_DELTA.md` states that the X₃₅ kill binds first and is the stricter
of the two. It is discharged here at the oracle's scope, which is what lets a
floor comparison mean anything — and it re-arms the moment `z_model` changes.

---

## 7. `CEQ_V15_2_DELTA.md` — what this node does and does not deliver

**(d) Cramér–Rao localization floor — NOT COMPUTED, and no number is quoted.**

The delta's kill is explicit: *any CRB number quoted before its must-fire is
STRUCK*. None is quoted here, and the reason is structural rather than a
shortage of time, so it is worth stating for whoever builds the CRB node:

> The onset in this bed is a **discrete index**. It enters the likelihood only
> through `t* ∈ {0, …, n−1}`; the likelihood is not differentiable in it, the
> score does not exist, and there is therefore no Fisher information and no
> Cramér–Rao bound to be at a distance from. This is not a property of the
> estimator — it is the parameterization. The prior art the delta cites
> (Cramér–Rao bounds for delay estimation) is for a **continuous** delay
> parameter, and it does not transfer to a change-point index without one.

The consequence for the CRB node: **a CRB-bearing localization bed needs a
continuous onset parameter** — a switch-on with a continuous time, or a ramp
whose onset is a real number — and `bed_k`'s plant field takes an integer index
today. The classical result for the discrete case is not a CRB at all (the
change-point estimator does not converge at `√n` and its limit is an argmin of a
random walk), so a "CRB" fitted to the present bed would be a fabricated floor,
which is the thing the delta strikes.

What is reported instead, and is not a floor: the **measured** interval of onset
offsets, `[0, 1]` at 2.5/97.5 percentiles over 500 seeds at `sd = 0.05`
(§4). That is a CI. Distance-to-CRB is left blank rather than estimated.

**(a)/(b) competing estimators — not built here, and made consumable.**
`run_detect` returns `residual` and `energy` **as whole fields**, and
`run_residual` returns `(bed, r)`, so the exact source solve `(I − A)r` and the
time-reversal `Wᵀr` score the identical field this onset was read from rather
than a separately generated one. `gate_estimates` is the per-index drive in the
form X₃₅b's re-propagation consumes.

**(2) superposition — added, cheaply.** `plant` accepts a list of sources and
sums them; §4 carries the two-source reading. It cost no redesign.

**(c) Kramers–Kronig — not touched.** Out of scope for this node; no KK number
is cited anywhere in it.

---

## 8. THE BED VARIANT — one field in `ceq/beds/bed_k.py`

`build(kind, n, seed, plant=None, **params)`. `plant` is `{index, magnitude}`,
a list of them, or `None`; the manifest gains exactly one key:

```
plant = {present, index, magnitude, indices, magnitudes, u}
```

- Written on **every** manifest, planted or not (`present=False`, `index=None`),
  so a run can never be scored against the wrong ground truth by a caller who
  forgot to ask.
- `index`/`magnitude` name the **earliest** source — the only one an onset
  detector can call; `indices`/`magnitudes` carry all of them for the
  source-recovery estimators that can separate them.
- `u` is the latent's realisation, recorded as ground truth **for scoring
  only**. It influences `z` and is absent from every visible input the manifest
  hands an arm (`b`, `pos`). A detector that reads `bed["plant"]` is scoring
  itself; `oracle_visible` reads `b` and the kernel and nothing else.
- Seeding follows `bed_k`'s existing discipline: one rng, drawn **after** `b`,
  so `b` is bitwise identical at a given seed whether or not a plant is present
  — planted and unplanted runs are matched pairs, not independent draws. Each
  source draws a full-length vector and masks the prefix, so moving one
  source's index does not reshuffle another's realisation.

`python -m pytest tests/beds -q` → **15 passed**, the same 15 as before the
change.

---

## 9. LIMITS

The visible model is the oracle and nothing is trained, so every flatness
number above is a statement about the observation model rather than about any
arm, and re-arms the X₃₅ kill when `z_model` changes. The plant enters `z`
directly at its own index rather than through the memory kernel; a latent
driven *through* `K` would shift its own onset by the kernel's delay and is a
separate measurement. The onset offsets at `sd = 0.05` have a tail beyond ±1
(5 of 500 at +2), so the ±1 claim is 99.00% at that noise, not universal, and
is stated with its noise level per the delta's second kill. `magnitude` is
biased up by the noise variance inside the root. All numbers are single-machine,
float64, at the seed blocks named in each section; no CRB, KK, or CUSUM number
appears anywhere in this node.
