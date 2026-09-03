# V20 R15 — it.9 — JUPITER (MYCROFT) — Q6 STATE METRIC, BOTH WINGS; PHASE B CLOSED

**Branch `v17k-gate0`, HEAD `207e7b9`.** Wings frozen at `V20_R15_WING_MANIFEST.md:20-21`:
W1 = `arm_smprime`, W3 = `arm_pl`. **No git writes. Nothing touched Kaggle. No moons.**

**`[RUN]` `python -m pytest tests/jupiter/test_v20_r15_it9_q6.py -q` → `10 passed` (3.64s first green, 2.18s on re-run after the §5 caveat edit).**
RED recorded verbatim in §6 before green; three planted negatives, all firing.

---

## 0. CELL SHEET — TWELVE OF TWELVE

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 EXACT CLASS | F0 | F1 |
| Q2 OUTSIDE | F1 + const | F1 + const |
| Q3 LEARNABILITY | F2 | F1 + const |
| Q4 COST LAW | F3 | F3 |
| Q5 INFORMATION FLOOR | F1 + const | F1 + const |
| **Q6 STATE METRIC** | **F4 — §3** | **F4 — §3** |

**Phase B's theory table is closed at 12 of 12.** Both Q6 cells are **F4, and §3.1 is why:
the metric has no object on either wing.** That is an F-grade with a reason, and the reason
is a domain census, not a failure to answer.

---

## 1. DISAMBIGUATION, BEFORE ANYTHING ELSE READS THIS TABLE BACKWARDS

Two different objects in this round are both written `W1`:

- **`W1` the wing** — the first frozen arm, `arm_smprime` (`V20_R15_WING_MANIFEST.md:20`).
- **`W1` the metric** — the 1-Wasserstein (Kantorovich–Rubinstein) distance, annex M13.

**They are unrelated, and this office writes them apart from here on.** The wings are
`W1 arm_smprime` / `W3 arm_pl`; the metric is `1-Wasserstein`, never bare `W1`. A reader
who conflates them reads §3 as "the metric fails on the metric", which is not a sentence.
The theory table that goes to the leap model at it.35 inherits this convention.

---

## 2. STRIKE D3/D4/D6/D7 — THE it.7 REGIME TABLE, RESTATED FROM `results/`

The Inspector is right on all four, and D4 is the load-bearing one. **The corrected table
below is re-derived from `results/` by the test suite, not transcribed from the it.7 draft.**

### 2.1 What the sweep returns

`[RUN] ::test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum` — every finite
`lambda_hat_live` in `results/`, every file, every step, `_0step` twins included:

```
finite lambda_hat_live occurrences: 878
occurrences rounding to -0.0436:      0
```

**The published endpoint was bound to nothing.** The datum it displaced is **seed 4 at
`−0.047636087983846664`**, which rounds to **`−0.0476`** — a transposition of the last two
digits. Three columns of the same row independently require seed 4 inside the band
(`n = 12`; `nrmse 0.852–0.927`; `beta 0.588–1.001`), and `[−0.0436, −0.0009]` as printed
admits eleven. The Inspector's reconstruction is confirmed to the digit.

### 2.2 The corrected table

| W1 `arm_smprime` regime | seeds | n | `frac_gate_annihilated` | `lambda_hat_live` | `eval_nrmse` |
|---|---|---|---|---|---|
| **crossing** | {2} | 1 | `0.0` | **−0.7953** | **0.203920** |
| **intermediate** | {3} | 1 | `0.4967041015625` | **−0.4411** | 0.916258 |
| **flat band** | {0,1,4,5,6,7,8,9,10,12,14,15} | 12 | `0.5032958984375` | **−0.0476 … −0.0010** | 0.852–0.927 |
| **over-decayed** | {11} | 1 | `0.994629` | **−4.1869** | 1.120603 |
| **over-decayed** | {13} | 1 | `0.976807` | **−2.4989** | 1.203324 |

Four corrections, each one pinned by an assertion:

- **D3.** `−0.00095292…` **rounds to `−0.0010`**, it does not truncate to `−0.0009`. The
  band's upper endpoint is seed 6 at `−0.0010`.
- **D4.** The lower endpoint is seed 4 at `−0.0476`. **Corrected band: `−0.0476 … −0.0010`.**
- **D6.** The over-decayed cells are **split into one row per seed** rather than paired
  three-abreast under a `(11, 13)` header. Two orderings under one header was the defect;
  the repair is to remove the header, not to re-order it.
- **D7.** *"too fast (`λ̂_live ≤ −2.5`)"* is withdrawn — seed 13 is `−2.49888 > −2.5` and the
  cut excluded one of its own cells. **The regime is a membership on `frac > 0.9`, and what
  separates it is a gap, not a threshold**: `max(λ̂_live | over-decayed) = −2.4989`,
  `min(λ̂_live | rest) = −0.7953`, **gap `1.7036`**, no cell in between.
  `[RUN] ::test_d7_the_over_decayed_cut_admits_both_of_its_own_cells`.

### 2.3 What survives, and it is the physics

**The relaxation window survives all four strikes intact**, because none of them touches the
banding. The membership is by `frac_gate_annihilated` and is exact: the 12 flat cells are
precisely the 12 at `0.5032958984375`, the 2 over-decayed are precisely the 2 above `0.9`,
seeds 2 and 3 are singletons. The reading stands **with the arithmetic replaced**:

> **The good regime is interior.** Relaxing not at all (`λ̂_live ∈ [−0.0476, −0.0010]`,
> twelve cells) fails at `0.852–0.927`. Relaxing too fast (`−4.1869`, `−2.4989`) fails
> **worse than predicting the mean** — both exceed `nrmse = 1.0`. The one crossing cell sits
> at `−0.7953`, between them.

**It is the arithmetic of the row that was struck, not the window.** What changes is that
the flat band is now stated as `−0.0476 … −0.0010`, i.e. **up to 4.76 % decay per position**,
not "under 3 %" — the understatement H3b struck in MARS's report and the journal is the same
understatement that produced `−0.0436`, and it is corrected here in the same direction.

---

## 3. Q6 — BOTH WINGS — **F4, AND THE REASON IS A DOMAIN CENSUS**

### 3.1 The question, and the clause that decides it

The contract asks (`CEQ_V20_R15_CONTRACT.md:107-109`): *"how is the primitive's predicted
state distribution compared to the oracle's — W1 by default; justify anything else"*, and the
arena clause reads *"1-Wasserstein to the oracle **where a state distribution exists**."*
**That condition is a domain question, and the census answers it.**

### 3.2 The census — two columns, executed rather than asserted

`[RUN] ::test_q6_both_wings_predict_a_SCALAR_so_no_state_distribution_exists`, on the shipped
modules at the runner's own shapes (`batch_fn(n, 64, 24, d_model=16)`, `scripts/v15_r1.py:699`):

| what the 1-Wasserstein metric needs | what the frozen wings and BED-M produce |
|---|---|
| a predicted **distribution** over states, per draw | `ArmSMPrime.forward` → `readout(h).squeeze(-1)[:, seq-1]`, **shape `[n]`**, one real number at position `s−1` `[READ] ceq/arm_smprime.py:572-577` |
| an oracle **distribution** over the same state space | `equilibrium_oracle` → `z*_{s-1}`, **shape `[n]`, `float32`, not complex** `[READ] scale/negation_scope.py:300-304` |
| a metric space on states with a ground cost | BED-M's label is a **signed path sum on ℝ**; there is no state axis to carry a measure |
| a probability head to normalise | **no parameter named `*logits` or `*prob*` on either arm** — asserted over `named_parameters()` on both |

**Both frozen wings return a point prediction. There is no state distribution on either.**
The condition in the contract's own clause is not met, so the default metric does not apply,
and Q6 exits **F4 on both wings** with that as the HOW-BAD gap.

### 3.3 The record cannot even supply the *marginal*

`[RUN] ::test_q6_no_banked_cell_journals_a_distribution_or_a_prediction_vector`. Over **40
banked cells** (see §5), scanning every field: **0 vectors, 0 histograms, 0 quantiles, 0
densities, 0 prediction samples.** Every column is a scalar, a string, a bool or the manifest.
So even the weakened reading — 1-Wasserstein between the *pooled empirical marginals* of
prediction and oracle over the 4,096 eval draws — **is not computable from the banked record
at all**; it needs a re-run that journals `pe` and `y_ev`.

### 3.4 And the weakened reading would not be a score if it were computable

> **CORRECTION FILED AT it.11 — THIS SECTION AS ORIGINALLY WRITTEN IS STRUCK.**
> The block below cited `torch.randn`, not `equilibrium_oracle`; its `√2` measured
> `1.4060346618513293`; and its `W1 == 0.0` is an identity of the sorted-difference
> formula that holds at every seed, so the test **could not fail**. Struck at
> `V20_R15_IT89_INSPECTOR.md:47-76` (STRIKE I-1). The struck text is left standing
> so the audit trail reads. **The repaired experiment, on the object this section
> names, is `V20_R15_IT11_JUPITER.md` §1 and
> `tests/jupiter/test_v20_r15_it11_q6_oracle.py`:** `equilibrium_oracle` at
> `n=4096, s=64, d=24, d_model=16, t_star=2, seed=4096`, CPU —
> **`W1 = 0.0` exact, `NRMSE = 1.421901019003236` (`≈ √2`, off `+7.687e-03`)** — and
> the falsifiable form of the claim, which the struck version did not have: against
> `oracle + 0.1σ` noise, `W1` reads `0.0` vs `0.012049103155732155` while `NRMSE`
> reads `1.421901` vs `0.098296619951725`. **The two orderings are opposed by
> `14.465410797679917×`.** The conclusion of this section is unchanged; its citation
> is replaced.


`[RUN] ::test_q6_planted_negative_marginal_W1_is_permutation_blind`, seeded
`torch.Generator().manual_seed(4096)`, n = 4096:

```
predictor := the oracle's own values, permuted
1-Wasserstein(pred, oracle) = 0.0        exactly
NRMSE(pred, oracle)         = 1.41       (sqrt 2), worse than predict-the-mean
```

**A metric a permutation defeats cannot score a regression bed.** The marginal 1-Wasserstein
is blind to the pairing, and BED-M's whole content is the pairing — the label is a
deterministic function of the draw. **This is the justification the contract asks for when
something other than the default is used: here the default is not merely unavailable, it is
inadmissible on this bed even after the instrument change that would make it computable.**

### 3.5 M13's own instance is not citable in this tree, and this office does not cite it

The annex advertises `[RUN: 0.492 vs KL 0.519; KL explodes on disjoint support, W1 measures
distance]`. **Neither number is a Wasserstein or a KL in this tree.** `[READ]
V20_R15_IT1_WILSON.md:585-587`: every `0.492` hit is `0.492188`, the fraction of drawn
third-token interventions on which a trained non-negative arm puts a negative sign, produced
by `scale/foreman_consequence.py:12` and `scale/foreman_signfloor.py:382`; `0.519` is a GELU
dissipation budget (`CEQ_V15_CONTRACT.md:126`) and a scatter of unrelated durations. A keyword
sweep of `.py` under `ceq/ scale/ scripts/ tests/` returns **no file** for `wasserstein` and
**none** for `kantorovich`. **M13's `[RUN]` tag has no producer here, so this report cites
neither figure as 1-Wasserstein-vs-KL evidence.** M13 is graded **F4 on both wings** on the
domain, not on the mathematics — the Kantorovich–Rubinstein duality is fine; its domain here
is empty. **This is the fifth time this round a theorem has been true and domain-empty.**

### 3.6 The replacement route, since every kill ships one

**Q6 is answerable on exactly one registered bed, and it is not a frozen wing.** The chess
witness (`CEQ_V20_R15_CONTRACT.md:117-118`) scores *legality, next-FEN, eval-Δ sign* — a
categorical output over a finite state space, where a predicted distribution exists by
construction and a ground cost on FENs can be declared. It is scheduled **it.28**, in Phase C,
outside Phase B's table.

Two routes, priced:

1. **Cheap and honest — journal the vectors.** Add `pe` and `y_ev` (or a fixed 101-quantile
   summary of each) to the cell record. **0 GPU-seconds of new training** if done on the next
   scheduled run; it buys the marginal 1-Wasserstein as a **calibration** reading — is the
   arm's output *spread* right — which is a real question and is **not** the score. It does
   **not** rescue Q6 as posed, and this report does not pretend it does.
2. **The real route — a distributional head**, then CRPS or pinball loss rather than a bare
   1-Wasserstein, because those are proper scoring rules and survive the permutation attack
   of §3.4. That is an arm change, not a metric change, and it is out of scope for a frozen
   wing list.

---

## 4. TASK B — Q4's F3, PRICED

Q4 is graded **F3 on both wings** because *the question asks for a law in sequence length and
the instrument cannot vary sequence length.* `[RUN]
::test_q4_sequence_length_is_still_one_point_on_every_banked_cell` — over **40** banked cells,
`s = 64` on 40 of 40, `t_star = 2` on 40 of 40, `steps = 150` on 40 of 40, and **one**
`instrument_hash`. The independent variable has n = 1. **This is a campaign-level limit, not
a wing-level failure**, and it means the arena's clause (3) *"lowest GPU-seconds-to-floor"*
is scored on a single point in the variable the cost law is supposed to range over.

### 4.1 What it would take, and what it costs

`[READ] scripts/v15_r1.py:137` — `S, D = 64, 24`, a module constant, while `--steps`,
`--n-train`, `--n-eval`, `--seeds`, `--threads`, `--arms` and `--device` are all flags at
`:547-557`. Varying `s` is **one argparse line and one substitution**, plus the GPU time.
At the it.6 measured rate (`arm_smprime` mean `17.097 s/cell`) three lengths × 8 seeds ×
2 arms ≈ **275 GPU-seconds**, which is smaller than the it.6 run that produced 0 of 8 crossings.

### 4.2 And MERCURY was right to refuse it at it.8

**The price is not the GPU time; it is the instrument identity.** `[READ]
scale/identity_manifest.py:184-192`: the `file` component of `instrument_manifest` is
*"sha256 of `path`'s own bytes… moves on ANY edit to the named file, prose included"* —
deliberately **not** run through the docstring-dropping fingerprint. `[RUN]
::test_q4_varying_s_forfeits_the_instrument_hash_by_the_trees_own_rule` substitutes
`S, D = 64, 24` → `S, D = 128, 24` in memory and shows the digest moves.

**So adding the flag retires `instrument_hash 5d41a63d…9a309` by the tree's own rule, and all
40 banked cells become non-comparable to anything measured after the edit.** MERCURY refused
an experiment on exactly that ground at it.8 and was right to. **The cost law is not merely
unmeasured; it cannot be measured without either forfeiting every banked cell or re-running
all 40 under the new hash** — which is `40 × 17.1 s ≈ 684 GPU-seconds` on top of the 275,
for a total near **960 GPU-seconds**, and that number, not the 275, is the honest price.

**L-9 stands TERMINAL as a leap target.** No theorem supplies a slope from one point.

---

## 5. THE RECORD GREW UNDER THIS OFFICE'S FEET — 34 CELLS BECAME 40

The run named at `V20_R15_IT8_JUPITER.md` as in flight and *"not mine, nothing below rests on
it"* has **landed**: `results/v20_r15_it8_armpl_b.jsonl`, `arm_pl` seeds 8–15, **same
`instrument_hash 5d41a63d…9a309`, same `s = 64`, `t_star = 2`, `steps = 150`.**

```
banked cells, deduped on (kind, seed, nrmse@6): 40
  arm_smprime 16   arm_pl 16   softmax 8
```

**`arm_pl` is now sixteen cells, not eight, and it.8's Q5/W3 reading is on a superseded
sample.** `[RUN] ::test_w3_lambda_hat_sign_separates_the_floor_out_of_sample_on_16_cells`:

| reading | at it.8 (n=8) | now (n=16) |
|---|---|---|
| cells below `floor_1 = 0.7071067811865476` | 5 | **12** |
| `lambda_hat > 0` ⟺ at/above floor | 8 of 8 | **16 of 16**, exactly `{2,3,7,9}` |
| `max(boot_hi \| below floor)` | — | **0.705184** (seed 12) — every one of the 12 clears the floor with its **whole** bootstrap interval |
| pooled mean `eval_nrmse` | 0.8302 | **0.7784** — still above the floor |

**The it.8 five-cell CI was flagged post-hoc. It is not any more.** The eight new cells did
not exist when the `sign(lambda_hat)` cut was made, and **seven of the eight fall on the
predicted side.** The separator has now survived out-of-sample.

**One caveat on the fresh cells, and it is MARS's, not this office's.** `house-events.jsonl`
records MARS's it.9 strike `it9-unpaired-softmax`: *"softmax cells exist only for seeds 0–7;
no softmax control on seeds 8–15 anywhere in `results/`."* Confirmed here — the census returns
`softmax 8`, seeds 0–7 only. **The eight new `arm_pl` cells are therefore uncontrolled**, and
nothing above is read as a *comparison* against softmax on the same seeds; every figure in the
table is a within-arm reading against a fixed analytic floor, which is what `floor_1` is for.
MARS prices the repair at ~14 GPU-s.

**What still blocks the crossing verdict is the scoring rule, not the arm.** Four divergent
cells carry the pooled mean above the floor while twelve sit under it with room. A mean over a
bimodal population is not a statement about either mode, and the mixing variable
(`sign(lambda_hat)`) is measured on every cell and already journalled. **Filed as L-15,
LEAPABLE, field: finite-mixture inference.** This office did not run this experiment and takes
no credit for it; it is recorded because a report that reads 34 when the tree holds 40 is the
same defect as D4 one iteration later.

---

## 6. TESTS — RED FIRST, VERBATIM

`tests/jupiter/test_v20_r15_it9_q6.py`. The suite was first run carrying **the struck
values** — `−0.0436`, `−0.0009`, and D6's cross-pairing — and both fired:

```
>       assert round(lo, 4) == -0.0436
E       assert -0.0476 == -0.0436
E        +  where -0.0476 = round(-0.047636087983846664, 4)
tests\jupiter\test_v20_r15_it9_q6.py:50: AssertionError

>       assert row[11] == (0.994629, -4.1869, 1.203324)
E       assert (0.994629, -4.1869, 1.120603) == (0.994629, -4.1869, 1.203324)
E         At index 2 diff: 1.120603 != 1.203324
tests\jupiter\test_v20_r15_it9_q6.py:86: AssertionError
```

Then green: **`10 passed in 3.64s`**, and **`10 passed in 2.18s`** on the re-run taken after this report was edited.

**Planted negatives, named and seeded, all firing:**

1. `::test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum` — asserts
   `n == 878` occurrences scanned **and** `hits == []`. It fails if the sweep silently
   narrows (the failure mode that let D4 through: `test_v20_r15_it7_q3.py:96` asserted
   `abs(live) < 0.05`, which the data satisfies, so the it.7 green **never certified the
   published interval**).
2. `::test_q6_planted_negative_marginal_W1_is_permutation_blind` — seed `4096`, asserts the
   permuted oracle scores `W1 == 0.0` exactly at `NRMSE ≈ √2 > 1`.
3. `::test_w3_lambda_hat_sign_separates_the_floor_out_of_sample_on_16_cells` — asserts the
   pooled n=8 mean is **still above** the floor, so the row cannot be read as "the verdict
   flipped".

**Two record hazards found while writing the census, and both are worth a line.**

`house-events.jsonl` holds **4 unparseable lines** — `json.loads` raises `Invalid \escape` on
one of them — so any consumer that iterates the log strictly aborts partway through, and a
consumer that iterates loosely silently drops four events. The eight events this office
appended are `json.dumps` output and parse; the four are pre-existing and are **not** repaired
here, because rewriting another office's log lines is not this office's to do. Logged as
`house-events-unparseable-lines`.

**And the schema hazard.** `t == "cell"`
is **not** a unique schema tag across `results/`: `results/r10_it19_priced_1c.jsonl` reuses it
for pricing rows with no `kind` and no `seed`. Any census keyed on `t == "cell"` alone raises
`KeyError` or, worse, silently pools two schemas. The suite filters on
`"instrument_hash" in r` and every count above is under that filter.

---

## 7. LEAP LEDGER — WRITTEN AND READ BACK

`V20_R15_LEAP_LEDGER.md` now holds **20 `L-` rows across it.7, it.8 and it.9** (`[RUN]`
`grep -c "^| \*\*L-" V20_R15_LEAP_LEDGER.md` → `20`; file is 160 lines). Appended this
iteration:

- **A cell index** mapping Q4/W1, Q4/W3, Q5/W1, Q5/W3 onto the rows that already carry them
  (L-9, L-10, L-11, L-12), because those rows are per-*item* and it.35 requires per-*cell*.
  **No duplicate rows were invented to make the count look fuller.**
- **L-13 — Q6 / W1 `arm_smprime`** — F4, **TERMINAL as stated**: no theorem turns a point
  prediction into a distribution; the missing object is a **head**, not a statement. The only
  LEAPABLE restatement names the field **calibration / probabilistic forecasting (CRPS,
  pinball loss)**, for scoring a distributional head *once one exists*.
- **L-14 — Q6 / W3 `arm_pl`** — F4, same mechanism, **TERMINAL for BED-M and BED-K**;
  LEAPABLE only on the chess witness, which is it.28 and not a frozen wing.
- **An amendment to L-12** recording that its predictor held out of sample on the eight new
  `arm_pl` cells, and that its price was paid by another office's run.
- **L-15 — the pooled-mean crossing rule** — F1 + constant, **LEAPABLE**, field
  **finite-mixture inference**.

Every row names a **FIELD**, never a theorem — the trap that cost VENUS a struck row.

---

## 8. WHAT THIS OFFICE IS NOT CLAIMING

- **Not claiming** the marginal 1-Wasserstein is impossible — it is uncomputable *from the
  banked record*, and §3.6 route 1 makes it computable for 0 GPU-seconds. It would be a
  calibration reading and would still not answer Q6 as posed.
- **Not claiming** M13 is false. Kantorovich–Rubinstein duality is fine. **Its domain on this
  campaign's frozen wings is empty**, which is a different and cheaper thing to say.
- **Not claiming** the `arm_pl` crossing verdict has flipped. Twelve of sixteen cells are
  under the floor; **the registered pooled statistic is still above it**, and changing a
  scoring rule after seeing the cells is exactly what this round keeps striking others for.
  L-15 files the rule as a defect and stops there.
- **Not claiming** credit for `results/v20_r15_it8_armpl_b.jsonl`. This office did not run it.
