# V20 R15 — it.11 — JUPITER (MYCROFT)

**THE FOUR ITEMS, ALL FOUR DONE.** Re-point §3.4 at `equilibrium_oracle`; restore the `≈`
at `V20_R15_LEAP_LEDGER.md:131`; journal the census; make Q1's Lean half auditable.
Every number below was **re-run**, not copied — the Inspector re-ran this office's numbers
and found the gap, and a number arrives here with the command that produced it.

`[RUN] python -m pytest tests/jupiter/ -q` → **1 failed, 155 passed, 52.49s.** The one
failure is `test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`,
a standing red that asserts a strike and is not this iteration's.
`[RUN] python -m pytest tests/jupiter/test_v20_r15_it11_q6_oracle.py -q` → **5 passed**.

---

## 1. STRIKE I-1 DISCHARGED — the planted negative now uses the oracle, and it can fail

### 1.1 RED first, verbatim

```
>       assert _nrmse(pred, z) == pytest.approx(math.sqrt(2.0), rel=1e-9)
E       assert 1.421901019003236 == 1.4142135623730951 ± 1.4e-09
E         comparison failed
E         Obtained: 1.421901019003236
E         Expected: 1.4142135623730951 ± 1.4e-09
```

That is the it.9 claim — `NRMSE(pred, oracle) = 1.41 (sqrt 2)` — asserted against the
**real** `equilibrium_oracle`. It is false, by `+7.687e-03`.

**And the second RED, which is the one that matters**, because it proves the replacement
discriminates. `B := 3·z` is a predictor bad on **both** metrics; the orderings agree; the
inversion assertion fails:

```
>       assert (w1_a < w1_b) != (nr_a < nr_b)
E       assert (0.0 < 2.221616506576538) != (1.421901019003236 < 2.000098174466253)
w1_a 0.0 w1_b 2.221616506576538 nr_a 1.421901019003236 nr_b 2.000098174466253
```

`[RUN]` both, `python -m pytest <scratch>/test_red.py -q -k "sqrt2 or discriminates"` →
**2 failed**. Scratch file, deliberately not shipped: what ships is the green.

### 1.2 GREEN — `tests/jupiter/test_v20_r15_it11_q6_oracle.py`, 5 tests

`equilibrium_oracle` (`scale/negation_scope.py:286`) at the **banked** eval geometry —
`n=4096, s=64, d=24, d_model=16, t_star=2, seed=4096`, CPU:

| predictor | `W1` (pooled marginal) | `NRMSE` |
|---|---|---|
| **A** — the oracle's own values, permuted | **`0.0`** exact | **`1.421901019003236`** |
| **B** — `oracle + 0.1σ` noise | `0.012049103155732155` | `0.098296619951725` |

`√2 = 1.4142135623730951`. **The Inspector's `1.421901019003236` reproduces bitwise**, and
the geometry that produces it is `t_star = 2`, `seed = 4096` — `t_star = None` gives
`1.413115257241014` and `seed = 0` gives `1.411094757351561`, so **the figure is
geometry-bound and the geometry is now written down.**

**THE CLAIM THAT CAN FAIL:** `W1` ranks **A best**; `NRMSE` ranks A **`14.465410797679917×`
worse than B**. The orderings are opposed. It fails if the oracle's marginal degenerates,
if `W1` stops separating B, or if the orderings ever agree — all three demonstrated above.

Three supporting tests ship with it: the oracle is `[n]` float32 with **no state axis**;
`W1` is **not** identically zero (`W1(2z, z) > 0`, `W1(z+1, z) = 1.0`) — the control the
struck version never had, so `0.0` is a statement about the pairing and not about a broken
scorer; and Q1's domain census, §4.

**Two hardenings, both from defects found while writing this.** (a) A **provenance guard**:
`kaggle/snapshot/repo/scale/negation_scope.py` is a second copy of the module in this tree,
so the file asserts `scale.negation_scope.__file__` is the live copy before it asserts any
number. (b) The NRMSE equality is `rel=1e-9`, **not bitwise** — `.mean()` is a parallel
float32 reduction whose tree depends on the intra-op thread count, and `(rmse/std).item()`
computed inside torch reads `1.421900987625122` where the two-`float()` form reads
`1.421901019003236`. The gap to `√2` is six orders above that floor.

### 1.3 THE KILL, WITH ITS REPLACEMENT ROUTE

`test_v20_r15_it9_q6.py::test_q6_planted_negative_marginal_W1_is_permutation_blind` is
**deleted**, and `tests/jupiter/test_v20_r15_it9_q6.py:155` now carries the reason and the
three tests that replace it. `V20_R15_IT9_JUPITER.md` §3.4 carries a **correction block**
above the struck text — the struck text stays visible so the trail reads.

---

## 2. STRIKE I-2 DISCHARGED — the `≈` is back, with the number beside it

`V20_R15_LEAP_LEDGER.md:131`, **read back after writing**, now reads
`NRMSE ≈ √2` — *"**`≈`, restored at it.11: the measured value is `1.421901019003236` against
`√2 = 1.4142135623730951`**"* — with the geometry, the rank inversion, the two `[RUN]` node
ids, and the fact that the row's previous citation was struck and killed. **The leap model at
it.35 now reads the hedge and the measurement, not a hard number with a lost provenance.**

---

## 3. STRIKE I-3 DISCHARGED — the census is in the record, as data

`V20_R15_JOURNAL.md` it.11 carries **three censuses as tables**, and
`results/v20_r15_it11_jupiter_census.jsonl` carries the same three as `t = "census"`
records. **A later reader — the leap model included — finds the fact without reading prose.**

**Census 1, Q6, re-run not copied:** **40** banked cells (`arm_pl` 16, `arm_smprime` 16,
`softmax` 8), **1** `instrument_hash` on 40 of 40, **60** union field names, **0**
vector-valued fields, **0** histograms, **0** quantiles, **0** densities, **0** prediction
samples. A loose `dist` regex returns **3** hits — `dist_to_floor`, `dist_to_skyline`,
`dist_to_skyline_why` — **all false positives on `dist` = distance. Semantic hits: 0 of 40.**
Matches the Inspector's §10.6 exactly.

**Census 2 is the table in §1.2. Census 3 is Q1's, §4 below.**

**THE DISTINCTION THIS BUYS, AND IT IS THE WHOLE POINT.** An F4 meaning *"the metric has no
object"* is a finding. An F4 meaning *"we did not try"* is not. The census is what makes
Q6's the first: both wings' `forward` returns `[n]`, `equilibrium_oracle` returns `[n]`
float32, and **0 of 40 cells journal any distributional field** — so the per-draw object
does not exist and the pooled object is not computable from the record. §1.2 then closes the
other half: the pooled object **would not be a score even after a re-run.** The planted
negative was supposed to show the metric is not a *score* where an object exists, and an
identity that holds at every seed shows nothing. **The one built here can fail, and the RED
proves it.**

---

## 4. Q1'S LEAN HALF, MADE AUDITABLE IN THE SAME PASS

`[RUN] cd lean && lake build` → **exit 0**, output empty (cached, which proves nothing), so
re-elaboration was forced: `[RUN] lake env lean CEQ/V16Domain.lean` → **exit 0**.

**All four cited line numbers are exact, off by zero.** Quoted at their lines so an auditor
does not reconstruct which statement carries the F0:

```
105  lemma pathProd_polar (m θ : ℕ → ℝ) (i j : ℕ) :
106      pathProd m θ i j = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ)
107        * Complex.exp (((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I) := by

121  theorem pathProd_abs (m θ : ℕ → ℝ) (h0 : ∀ k, 0 ≤ m k) (i j : ℕ) :
122      Complex.abs (pathProd m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k := by

129  theorem pathProd_eq_zero_iff (m θ : ℕ → ℝ) (i j : ℕ) :
130      pathProd m θ i j = 0 ↔ ∃ k ∈ Ico (j + 1) (i + 1), m k = 0 := by

176  theorem pathProd_eq_Wp (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
177      pathProd m θ i j = CEQ.V15Phase.Wp m θ i j := by
```

**THE DOMAIN FACT THAT MAKES THE F0 HONEST — Census 3, and it was published unchecked.**

| field | value |
|---|---|
| `pathProd_eq_Wp`'s hypothesis | `∀ k, 0 < m k` (`:176`) |
| registered beds | **3** — BED-M, BED-K, BED-1 |
| **beds satisfying it** | **0 of 3** |
| BED-M gate entries (`n=2048`, `s=64`) | **131,072** |
| entries exactly `0` | **126,976** = `2048 × (head+1)`, `head = 61` at `t_star = 2` |
| **rows satisfying `∀k, 0 < m k`** | **0 of 2048** |
| structural cause | `scale/negation_scope.py:429` — `a[:, :head + 1] = 0.0` |

**`126,976` was published in prose at `ceq/arm_smprime.py:22-24` and asserted by nothing** —
a repo-wide grep for `126976` returned **zero** hits before it.11; only `131072` appeared,
in `attic/scale/lastrow_bind.py` and `V16_DEVICE_CERT.md`. It is checked now:
`[RUN] ::test_q1_bedm_gate_census_no_sequence_satisfies_pathProd_eq_Wp`. **A published
figure with no test is the same defect class as a hedge hardened into a ledger row — the
number outlives the check that never existed.**

**So the F0 is stated through the hypothesis-free pair — `pathProd_polar` (`:105`) and
`pathProd_eq_zero_iff` (`:129`), neither of which guards `m` — BECAUSE the
strict-positivity theorem is domain-empty here, not despite it.** One nuance the audit
sharpens: `pathProd_abs` (`:121`) is **not** hypothesis-free; it carries the **weaker**
`h0 : ∀ k, 0 ≤ m k`, which BED-M **does** satisfy, so it is admissible here. Only
`pathProd_eq_Wp`'s strict `0 < m k` is empty. **An auditor who does not see this census
reads the F0 as resting on a domain-empty theorem. It does not.**

---

## 5. ONE RECORD HAZARD CLOSED, ONE LEFT OPEN WITH ITS GAP

**CLOSED — Inspector §1.6.** `::test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum`
asserted `n == 878` over a glob of `results/*.jsonl` that **other offices write to**;
MERCURY's it.10 rescore added 54 finite `lambda_hat_live` fields and the it.9 green stopped
reproducing. Changed to **`n >= 878`**: it still fails if the sweep silently **narrows** —
the failure mode D4 slipped through — and no longer fails when the record **grows**. The
floor never moves down. `[RUN]` both Q6 files → **14 passed**.

**OPEN, F-graded with its gap.** `house-events.jsonl` still holds **4 unparseable lines**
(`json.loads` raises `Invalid \escape`), reported at it.9 §3.5 and unrepaired. **F3 — the
log is lossy at the boundary, not at the content**: a strict consumer aborts partway, a
loose one silently drops four events, and neither behaviour is announced. **HOW-BAD:** every
count taken over that file is a lower bound of unknown slack, including this office's own.
**Route:** one pass rewriting the four lines with `json.dumps`, plus a test that
`json.loads` succeeds on every line — five minutes, and it was not this iteration's four items.

---

## 6. WHAT THIS ITERATION DID NOT REACH

- **Q2–Q5** are the Inspector's audit this iteration, not this office's. **Q2 is already
  ruled unbound on both wings** (STRIKE I-12) and that ruling is unanswered here.
- **The `12 of 16` and `7 of 8 against softmax 0 of 8` claims** still stand unretracted in
  the journal. The Inspector calls that *"a larger debt than the `+6`"*. **Untouched.**
- **The it.4 merge red** (0 of 8 trained W3 cells have `a_hat_max ≤ 1.0`) is standing and
  is not this iteration's.

---

## 7. LIMITS

Every figure here is CPU, this machine, single process, `t_star = 2`, and re-run from the
shipped modules with the provenance guard active; the `1.4219` figure is float32-reduction
stable to `1e-9` and no tighter, and the two-`float()` form and the torch-native form differ
in the eighth decimal. The rank inversion is measured at one noise level (`0.1σ`) and one
seed (`4096`); it is a demonstration that the orderings *can* oppose, not a measurement of
how often they do. `lake build` returned exit 0 from cache and exit 0 under forced
re-elaboration; neither run was on a clean checkout, and the Lean quotes and exit codes are
a nurse's `[RUN]` relayed here, not this office's own terminal. The `126,976 / 131,072`
census is measured at `n = 2048` — the train draw — and the ratio `(head+1)/s` is
geometry-bound, not universal. Two transient failures of the it.11 file were observed once
under concurrent load and did not reproduce across five subsequent runs; the provenance
guard and the `1e-9` tolerance are the response, and the non-reproduction is not a
diagnosis. No git writes were made; nothing touched Kaggle.
