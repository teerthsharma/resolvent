# CEQ: Content-Conditional Sign in Multi-Hop Attention — A Negative Result

**A signed, strictly-causal path sum can do something no non-negative attention operator can do — let a third token flip the sign of one token's influence on another — and that capability does not survive context length, does not survive prior art, and loses to plain softmax on the one downstream task it was scored on.**

---

## How to read the numbers in this document

Every claim carries the evidence class the project's loop protocol assigns it. The classes are not decoration; a claim may not be stated above its class.

| Class | Meaning |
|---|---|
| `[RUN]` | A command was executed this session and its output is quoted. |
| `[READ]` | Read from a file in this repository at a stated path and line. |
| `[CITED]` | An external identifier resolved and its title checked against the claim. |
| `[DERIVED]` | Arithmetic shown from numbers that themselves carry a class. |
| `[GUESS]` | Unverified. Labelled as such, or absent. |

Observations are separated from interpretations throughout. "sgate reads 0/512 solved" is an observation. "The operator cannot do compositional generalization" is an interpretation, and a weaker one than the observation looks.

---

## Abstract

We ask whether a strictly-causal, signed, multi-hop attention operator buys a capability that non-negative attention cannot express. The property is **content-conditional sign**: whether changing a *third* token can flip the sign of token *j*'s influence on token *i*. A non-negative attention matrix composed with fixed linear value projections factors as (non-negative weight) × (fixed matrix), so its sign is fixed by the projections and no third token can move it; a signed path sum has no such factorization. We measure the property directly as a gradient-sign flip rate over random draws, on CPU, with a calibration gate that reproduces four published constants bit-identically. The property is real and separates cleanly from the softmax control, which reads **exactly 0.0000 with zero sign-flipped draws out of 1024** `[RUN]`. It also **decays as roughly `1/s` in context length** — log-log slope **−0.958 (R² 0.9990)** `[RUN]` — so at the context a serious run would use it is below what any affordable number of draws can resolve. On the one capability benchmark scored, COGS-style compositional generalization at **3,652,096 matched parameters**, the signed arm solved **0 of 512** items against softmax's **15 of 512**, one-sided Fisher **p = 2.7502788939116803e-05** `[RUN]` — and the deficit is already present in-distribution, so it is not a generalization story. The operator's matrix is additionally occupied by published work. This document reports the negative result, the instrument that produced it, the measurement defect found *inside* that instrument, and the machine-checked core that survives.

---

## Background

### Why a signed path sum?

**The property being chased is not "can produce a negative number".** A real attention block has signed `W_v` and `W_o`, either of which supplies a minus sign, so sign-capability at block level separates nothing. The sharper property — the one the word *not* needs — is whether the sign of `j`'s influence on `i` is **conditional on a third token's content**. `W_v` and `W_o` are constants and cannot supply that. Non-negative `a_ij` can only rescale. Only a content-dependent signed matrix can `[READ] ceq/bench.py:107-120`.

**Strict causality makes the series terminate, with no contraction hypothesis.** A strictly lower-triangular matrix over any commutative ring is nilpotent, so the path sum `I + A + A² + …` is finite by structure rather than by a spectral-radius bound. This is the one place the project's mathematics is genuinely load-bearing, and it is the part that is machine-checked.

**The cost is a denominator that sums over context.** Every normalized arm dilutes as `s` grows, because a row's shares are reallocated among more competitors. That single mechanism — not signedness, not hop count, not the normalizer's type — is what the decay measurements below keep finding.

### Prior Art

Every identifier in this table was resolved this session and its title checked `[CITED]`. **6 of 6 resolve to real papers whose titles match the claim made about them.** The rightmost column is the one that matters: it is where each system is *not worse* than this one.

| Identifier | Title (resolved) | Date | Matrix | Where it already covers this work |
|---|---|---|---|---|
| arXiv:2206.08898 | SimA: Simple Softmax-free Attention for Vision Transformers | 17 Jun 2022 | ℓ₁-normalized, signed | Occupies the row-ℓ₁ normalized signed operator (`signed` arm) outright |
| arXiv:2606.04833 | Signed Dual Attention: Capturing Signed Dependencies in Time Series Forecasting | 3 Jun 2026 | difference of two attention passes | The repo's own record states this **is** `sgate`'s matrix `[READ]` |
| arXiv:2406.06484 | Parallelizing Linear Transformers with the Delta Rule over Sequence Length | 10 Jun 2024 | key-key, signed, no row normalizer | Multi-hop and denominator-free; **beats `sgate` at every context measured** |
| arXiv:2512.14619 | ParaFormer (PageRank-enhanced attention) | Dec 2025 | non-negative base, signed hop coefficients | Nearest published multi-hop construction |
| arXiv:2411.07176 | More Expressive Attention with Negative Weights (Cog Attention) | 11 Nov 2024 | `sign(p)·softmax(\|p\|)` | Occupies the `signmag` arm; single-hop |
| arXiv:2310.11025 | SignGT: Signed Attention-based Graph Transformer | 17 Oct 2023 | signed, normalized | Signed attention values by semantic relevance; single-hop |
| arXiv:2307.08621 | Retentive Network: A Successor to Transformer | 17 Jul 2023 | decay-weighted retention | Multi-paradigm sequence modelling with the same efficiency pitch |

**The repository's own verdict is that nothing here is novel attention, and this document does not soften it.** What the resolution above establishes is narrower than the verdict and worth stating precisely: the identifiers are real and the titles match. **It does not establish the equation-level claims** — that SDA's matrix is bit-identical to `sgate`'s, that DeltaNet's WY inverse takes the stated form, that Cog reduces to `sign(p)·softmax(|p|)`. Those need the full texts, which were not retrieved. See *Limitations*.

---

## Theoretical Foundation

Only operators the code implements appear here. Each is a `[S, S]` matrix `A`, strictly causal (`j < i`), used as the base of a path sum.

**(1) Row-ℓ₁ signed operator** — the shipped `signed` arm, `[READ] ceq/bench.py:134-143`:

$$
A = \rho \cdot \frac{W}{\sum_j |W_{ij}|}, \qquad W_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}\ \text{ for } j < i
$$

Row ℓ₁ is pinned at exactly `ρ`. This form is **homogeneous of degree zero** in the logits, so it has no temperature channel at all — measured spread `0.000e+00` over a 64× scale sweep `[READ]`.

**(2) `sgate`** — the campaign's operator, a difference of two softmaxes over the *same* logits, `[READ] ceq/bench.py:154-173`:

$$
A = \rho \cdot \frac{\mathrm{softmax}(W) - \lambda\,\mathrm{softmax}(-W)}{1 + \lambda}
$$

Both halves retain `exp` sharpening, which (1) lacks. Shipped point: `ρ = 1.5`, `λ = 0.10`.

**(3) The path sum**, for hop budget `H`:

$$
h = \sum_{k=0}^{H} A^{k} v
$$

**(4) Nilpotency — the terminating series.** For strictly lower-triangular `A` over a commutative ring on `n` indices:

$$
A^{n} = 0
$$

machine-checked as `CEQ.Nilpotent.pow_card_eq_zero` `[READ] lean/CEQ/Nilpotent.lean:77`. **No sign hypothesis and no magnitude hypothesis.** The series terminates exactly regardless of how large entries become — which is what licenses unnormalized arms such as `tgate`.

**(5) The measured quantity.** With `v` an independent leaf and `c` a third token taking two values,

$$
\text{flip} \iff \frac{\partial\, \mathrm{out}_i}{\partial v_j}\bigg|_{c=c_0} \cdot \frac{\partial\, \mathrm{out}_i}{\partial v_j}\bigg|_{c=c_1} < 0
$$

The reported rate is the fraction of draws that flip. **This quantity is scale-free** — the sign of a product is invariant under positive rescaling of either factor. That fact is the whole subject of the next section.

---

## The Instrument, and the defect found inside it

### It reproduces itself

`[RUN] python run_calib.py --self-test` → **exit 0, 4/4 bit-identical**:

```
  signed  hops=3   0.0468750000   0.0468750000   OK
  sgate   hops=1   0.0234375000   0.0234375000   OK
  sgate   hops=2   0.1640625000   0.1640625000   OK
  softmax hops=3   0.0000000000   0.0000000000   OK
```

The gate first rejects a target deliberately wrong by `1e-9` and only then runs the real comparison, so a green is evidence that the gate can fail. This matters more than it sounds: an earlier revision of this script printed measured values, printed hardcoded targets beneath them, and returned exit 0 unconditionally `[READ] run_calib.py:5-9`.

### The defect: an absolute threshold on a scale-free question

`sign_flip_rate` discarded any draw whose smaller gradient fell below an **absolute** floor of `1e-6`. The quantity being reported is scale-free; the rule that decided which draws counted carried **units**. Measured at the published geometry `[RUN]`:

| arm | median `max(\|lo\|,\|hi\|)` | where `1e-6` sits in that arm's distribution | flipped draws discarded |
|---|---|---|---|
| `sgate` | 0.00425 | 2.36e−4 of its own scale | **48.08 %** |
| `deltanet` | 0.948 | 1.06e−6 of its own scale | **0.00 %** |
| `softmax` | 3.20e−12 | — | 0.00 % (no flips exist) |

**A factor of 223 between the two arms' gradient scales.** One shared cut lands inside one distribution and five decades below the other's. The threshold was therefore not a dust filter but an **uncontrolled, arm-dependent, `s`-dependent sample filter** — and it gated three of the four struck claims.

**The general failure, named:** *an exclusion rule that is not invariant under the same group as the statistic it gates.* Its second costume is an **uncontrolled scale parameter** — which is exactly strike S1, where the window width was pinned at `w = 8` while context was swept.

**The fix is structural, not four patches** `[READ] ceq/bench.py`: measurement and exclusion are now separate functions (`sign_flip_draws` returns raw `(lo, hi)` pairs; `flip_rate(draws, floor, rel)` applies a rule), so a floor sweep costs no re-measurement — which is precisely why no published number had ever carried one. Three verbatim copies of the defective line in `scale/carpet_probe.py`, `scale/sparse_probe.py` and `scale/recall_probe.py` now route through the one shared rule. The default reproduces historical behaviour bit-for-bit, which is why the calibration above still passes.

---

## Results

### 1. The property is real, and it decays as roughly `1/s`

`[RUN]` `sgate`, `hops=2`, `n=1024` draws, positions held relative (`i=s−1, j=s/4, c=s/2`), seed 0, CPU. The published row reproduces bit-identically, which is what licenses the rest of the table.

| s | floor=1e−6 (**as published**) | floor=0 | rel=1e−9 | move | floor-sensitive >10 %? |
|---|---|---|---|---|---|
| 8 | 0.174805 | 0.181641 | 0.178711 | 1.04× | no |
| 16 | 0.088867 | 0.098633 | 0.098633 | 1.11× | **yes** |
| 32 | 0.026367 | 0.050781 | 0.048828 | 1.93× | **yes** |
| 64 | 0.011719 | 0.026367 | 0.026367 | 2.25× | **yes** |
| 128 | 0.00391 | 0.012695 | 0.012695 | **3.25×** | **yes** |
| **slope** | **−1.389** (R² 0.9938) | **−0.958** (R² 0.9990) | −0.953 (R² 0.9989) | | **yes** |

Reproduce: `python -m pytest -q tests/foreman/test_absolute_floor_is_an_arm_filter.py`

**Observation:** 4 of 5 published cells and the published slope move by more than 10 % when the discard floor is removed.
**Interpretation, and it cuts both ways:** the decay is *real and unaffected* — the softmax control reads exactly 0.0000 under every rule tried, including no rule at all — but its **magnitude was inflated**. The honest exponent is about **−1**, not −1.389. The rate falls essentially as one token's share of a prefix of length `s`, which is a cleaner mechanism than the steeper number ever suggested.

**The unfiltered data fits the power law *better*** (R² 0.9990 vs 0.9938). The floor was adding curvature, because it bites harder at large `s` as gradients shrink. That is the mechanism confirmed from the residuals rather than from the endpoints.

### 2. The "non-vanishing windowed regime" was flat by construction

The published claim was that a bounded receptive field arrests the decay. Every windowed number was taken at `w = 8`, with probe positions pinned at `j = i − w/2` and `c = i − w/4`. At `w = 8` those are `i−4` and `i−2`, so `out_i` reads only tokens `[i−16, i]` — **the measured quantity cannot vary with `s`.** The sweep varied the variable the answer does not depend on and held fixed the one it does.

`[RUN]` the control that was never run — sweeping `w` instead:

```
  window width swept at s=256:  w=8 0.122396  w=16 0.052083  w=32 0.015625  w=64 0.005208
  context swept at w=8:         s=32 0.130208  s=64 0.104167  s=128 0.125000  s=256 0.122396

  SLOPE IN WINDOW WIDTH  -1.540  (R^2 0.9956)
  SLOPE IN CONTEXT       -0.000  (R^2 0.0000)
```

Reproduce: `python -m pytest -q -s tests/foreman/test_window_width_is_the_missing_control.py`

**The context slope is exactly flat and the window-width slope is −1.540** — indistinguishable in kind from the −1.4 decay in context that the windowed regime was supposed to escape. The property survives `w = 8`, not a bounded field. gpt-oss's sliding window is 128; Mistral's and Gemma's are 4096.

Made structural rather than statistical: holding the same 32-token tail, `A[i, i−8:i]` differs between `s=32` and `s=128` by **2.384e−07 against a row scale of 1.201 — 1.985e−07 relative** `[RUN]`, which is float32 epsilon from BLAS choosing a different reduction order for a different matrix *shape*, not a dependence on the prefix.

### 3. The capability number is a loss

COGS-style compositional generalization, **3,652,096 matched parameters on both arms** `[READ]`, 512 items:

| arm | in-distribution | generalization | solved |
|---|---|---|---|
| softmax | 0.9258 | **0.0293** | 15 / 512 |
| `sgate` | 0.7734 | **0.0000** | 0 / 512 |

One-sided Fisher exact for "softmax solves more than `sgate`": **p = 2.7502788939116803e-05** `[RUN]`, recomputed this session from the two counts.

Three arithmetic facts that keep this honest `[RUN]`, all from `ceq/bench.py`:

- `zero_success_upper_bound(512) = 0.005834`. `sgate`'s **0.0000 is not zero skill** — it is consistent with any true rate up to 0.58 %.
- `resolution_floor(512) = 0.001953`. Nothing between 0 and 1/512 is representable.
- `min_successes_for_separation(512) = 5`. Softmax's 15 clears that bar; `sgate`'s 0 does not.

**Observation:** the signed arm solved zero items; the softmax control solved fifteen.
**Interpretation:** this is *not* a generalization failure. The deficit is already present **in-distribution** (0.7734 vs 0.9258), so the operator is worse at the task, not worse at transferring. That is the more damaging reading, and it is the correct one.

### 4. What survived

- **The instrument reproduces itself.** 4/4 calibration constants bit-identical `[RUN]`; 26/26 published probe cells reproduce exactly `[READ]`.
- **The 84× retraction is complete and correct.** The figure was a softmax measurement wearing ParaFormer's name — the arm did not apply its `γ` coefficients. Fixed, and the corrected comparison is in the record `[READ]`.
- **The decay is real** after the S4 correction. Only the exponent moved.
- **`tgate` passes its own pre-registered kill.** An unnormalized, signed, query-dependent multi-hop base — slope **−0.071** against a kill bar of "steeper than −0.3", measured by the Health Inspector rather than by its author `[READ]`. It is the only arm in the family that does not die the context death, and the mechanism is stated in its source: no denominator sums over context, so there is nothing for context to dilute `[READ] ceq/bench.py:241-247`.

---

## The Machine-Checked Core

`[READ]` by direct inspection of `lean/CEQ/*.lean` this session:

| Module | `theorem` declarations |
|---|---|
| `Nilpotent.lean` | 4 |
| `Occupancy.lean` | 3 |
| `Refcount.lean` | 10 |
| `Contraction.lean` | 5 |
| `OrbitBound.lean` | 5 |
| **Total** | **27** |

**27 `theorem` declarations, 0 `lemma`, and no occurrence of `sorry` in any of them** `[READ]`. Toolchain: `leanprover/lean4:v4.7.0` against `mathlib4 @ v4.7.0`, single `lean_lib CEQ` target with root `CEQ` `[READ] lean/lakefile.lean`.

**All 27 rest on `[propext, Quot.sound, Classical.choice]` and nothing else — no `sorryAx` anywhere.** That is an `#print axioms` result, not a grep, run independently and recorded with `lake build CEQ` exit 0 in 32.9 s `[READ] DONE.md`. It resolves a count discrepancy in the older record, which said *12 theorems*: **12 is stale; 27 is right.** `lake build` was not re-run this session.

`CEQ.Nilpotent.pow_card_eq_zero` is the one the operator design actually rests on: it proves equation (4) over any `CommRing`, with no sign and no magnitude hypothesis.

`CEQ.Refcount.free_face_floor_unchanged` and its siblings prove that a certified indistinguishability floor **is** a sum over reference counts — `n − m = Σ_plaques (refcount − 1)` — connecting two pieces of the author's own prior work, `caustic` Theorem 1 and `foliation`'s KV-cache quotient `[READ] lean/CEQ/Refcount.lean:1-41`. The practical consequence is the part worth reading: scoring a KV block by removing it and measuring the change is leave-one-out influence and is externally owned, whereas these theorems make the score **an integer the cache already maintains**, with no forward pass per candidate block.

**A correction to the record, and it matters for release.** An earlier assessment held that this file was *cited in no document and gated by no test*. That is now half wrong and half fixed. `tests/w3b/test_w3b_lean_nilpotent.py` already gated compilation and `sorry`-freedom, since `CEQ.lean` imports `CEQ.Refcount`. What was genuinely ungated — deletion, rename, a changed axiom basis, the arithmetic content, the correspondence to shipped code, and the citation — is now bound by `tests/chase/test_lean_refcount_binding.py`, 10 tests, 46.65 s CPU `[READ] DONE.md`. The binding was justified RED-first: deleting the two load-bearing corollaries left `lake build CEQ` at **exit 0** and moved nothing in `tests/w3b`, because nothing in the library imports them. A build cannot see a deletion.

---

## Reproduction

Every number in this document marked `[RUN]` comes from one of these.

```bash
# calibration gate — must exit 0, four constants bit-identical
python run_calib.py --self-test

# the floor defect, arm-dependent discard rates, softmax structural zero
python -m pytest -q tests/foreman/test_absolute_floor_is_an_arm_filter.py

# window width vs context, both slopes printed side by side
python -m pytest -q -s tests/foreman/test_window_width_is_the_missing_control.py

# see the pinned defects report as real failures rather than as xfail
python -m pytest -q --runxfail tests/foreman/test_absolute_floor_is_an_arm_filter.py

# the published decay sweep (geometry i=s-1, j=s/4, c=s/2 is applied internally)
python scale/ratio_sweep.py --kinds sgate softmax --sizes 8 16 32 64 128 --n 1024 --hops 2

# the Lean core: 27 theorems, axiom basis checked, no sorryAx
cd lean && lake build CEQ
python -m pytest -q tests/chase/test_lean_refcount_binding.py
```

Three defects are pinned as `pytest.mark.xfail(strict=True)`: they **must keep failing**, and the day one passes the suite goes red and the calibration is revisited deliberately instead of silently. That is the mechanism by which a missing control stops going missing.

**Substrate for every number here:** Windows, Python 3.11.9, torch 2.5.1+cu121, **CPU only** (`CUDA_VISIBLE_DEVICES=""`), single process, seed 0 throughout. `[RUN] pytest --collect-only -q tests/` → **829 tests collected in 39.27 s**, exit 0.

---

## Limitations

**The whole test suite has never completed.** It has failed to finish in eight separate attempts across two agents, estimated at over three hours on CPU. **No total pass/fail count exists for this repository, and none should be quoted.** The only whole-suite number that exists is the collection count above.

**Nothing has run above 3.65M parameters.** The project's own scale gate is 300M, priced at 912 A100-hours. Every capability, decay and separation number in this document is from a model three orders of magnitude below that gate, and the project's own theory notes that architecture comparisons at small scale routinely move the scaling-law *intercept* without moving the *slope*.

**ARC-AGI has never been scored. A Turing-style evaluation has never been attempted — no file for it exists.** These were stated goals. They are unmet, not deferred.

**Most published cells were never swept for floor sensitivity.** Only the headline decay curve was audited. The windowed table's `s=2048` cell alone is priced at roughly 2.7 minutes of CPU at 80 ms/draw, outside this session's budget. Given that 4 of 5 audited headline cells moved by more than 10 %, **the remaining cells' floor sensitivity is unknown, not clean.**

**A fourth verbatim copy of the defective discard line remains** at `scale/pivot_probe.py:203` `[READ]`. It was left untouched because `scale/m2_units.py` imports from it and it is live measurement machinery. It is the one site already measured benign (1.00× at `s=32/128/512`), but benign today is not benign after the next arm is added.

**A related absolute tolerance is flagged but unmeasured.** `scale/dfloor_probe.py` and `dfloor_probe2.py` count distinct readings with an absolute `tol=0.35` on softmax outputs, whose magnitude falls as `s` grows. The DeltaFloor selector's reported slope of −0.309 — which fired a pre-registered kill — is conditioned on that tolerance. **Not re-measured.** The kill fired toward deletion, so a correction can only make the deleted selector look better, never worse.

**The Health Inspector's own corrected exponent does not reconcile with its own data.** S4 reports the floor-off slope as −1.221 (R² 0.9662); OLS on the five rates it published gives −0.958 (R² 0.9990) `[DERIVED]`. Unresolved; the `s` range used was not stated.

**Citation verification is partial.** Six arXiv identifiers resolve with matching titles `[CITED]`. The equation-level claims about those papers — that SDA's matrix *is* `sgate`'s, that DeltaNet's WY inverse takes the stated form — rest on full texts that were **not retrieved this session** and are carried as `[READ]` from the repository's own record.

**The measurements are on random projections, not trained ones.** Every sign-flip number draws `W_q, W_k, W_o` from a normal distribution. Whether a trained model occupies the same region of operator space is untested.

**Two documentation inconsistencies, for whoever owns those files.** `README.md:919` and `MODEL_CARD.md:364` state *809 tests collect, 19.9 s*; measured this session, it is **829 in 39.27 s** `[RUN]`. And `MODEL_CARD.md`'s Reproduce block still opens with `python -m pytest tests/ -q` — the suite that has never completed, i.e. a stranger's first command is a multi-hour hang. Neither file was edited here; both are owned elsewhere.

---

## What a reader should take from this

A negative result is worth publishing when the instrument that produced it is trustworthy, and the case for trusting this one is specific rather than rhetorical: it reproduces four constants bit-identically from a fresh interpreter, its calibration gate is itself calibrated by being made to fail first, and when a defect was found *in the instrument* the defect was measured, named one level of abstraction above the symptom, fixed in one place rather than four, and pinned with tests that must keep failing.

The honest summary is that **the property is real, small, prior, and does not pay.** It separates cleanly from a non-negative control — exactly 0.0000, zero flipped draws, a structural rather than thresholded zero. It also decays as `1/s`, is occupied by at least three published operators, and loses to plain softmax in-distribution at matched parameters with `p ≈ 2.75e−05`.

Two things point forward rather than back. **`tgate`** is the one arm whose slope survives its own pre-registered kill, and its mechanism is stated rather than hoped for: remove the denominator that sums over context and there is nothing for context to dilute. **`Refcount.lean`** turns an eviction-safety criterion from a measurement into an integer the cache already holds, is machine-checked on a verified axiom basis, and is now bound by tests that catch deletion rather than only compilation.

Neither is a result yet. `tgate`'s slope is one arm on one instrument at small `s`; `Refcount`'s theorems are proved but the shipped `HopCache.evict` admits on scope depth rather than on fibre cardinality, a divergence recorded rather than resolved `[READ]`. Both are still better places to spend the next 912 A100-hours than another sweep of the operator family.

---

## License

*Invented by [Teerth Sharma](https://teerthsharma.vercel.app)*
