# MARS — CEQ v20 ROUND 15, PHASE A, it.1

MORIARTY, run early against the slot it.3 so the wing list is never frozen
around a renaming. Two strikes filed, each with a RED test that was RED against
the tree as it stands, each with its replacement route. Four attacks that did
NOT fire are reported at the foot, and one defect in an existing strike's own
evidence command is filed separately.

Tests: `tests/mars_v20/`. No git write was performed. Nothing touched Kaggle.

**NURSES: NONE.** The dispatch was refused — `Concurrent subagent limit
reached. You can run 20 subagents at once.` — the pool was saturated by the
other planets at it.1. Every command below was run in-process by this node and
is quoted verbatim with its output. Logged to `house-events.jsonl` as
`{"t":"finding","agent":"Mars","text":"nurse dispatch REFUSED..."}`.

---

## STRIKE 1 — `frac_gate_annihilated` is the eval draw's sign census, not an arm reading

### (i) The claim struck, and where it lives

`results/v17k_r4_retake.jsonl` publishes `frac_gate_annihilated` once per
trained cell, in the same record as `eval_nrmse`, for all eight `arm_smprime`
seeds and all eight `arm_pl` seeds. Its producer is `scripts/v15_r1.py:386` —

```python
frac_gate_annihilated=float((~fin).double().mean()),
```

— inside `gate_columns`, whose comment at `scripts/v15_r1.py:369` offers it as
an arm reading: *"the annihilated fraction beside it -- the zeros are never
quietly dropped into a mean that would read finite."* The same quantity times
the draw size is `n_zero_gates`, which `ceq/arm_smprime.py:95` places in
`SMP_FIELDS` and `ceq/arm_smprime.py:396-411` folds into the cell's identity
hash.

**The measured values.** `[RUN]`

```
arm_smprime:t2:n2048:seed0   n_zero_gates=  4123  beta=0.732560 qk=1.276559 g=1.319505  frac=0.5032958984375
arm_smprime:t2:n2048:seed1   n_zero_gates=  4123  beta=0.896975 qk=1.320569 g=1.234473  frac=0.5032958984375
arm_smprime:t2:n2048:seed2   n_zero_gates=     0  beta=1.343933 qk=0.500633 g=1.079639  frac=0.0
arm_smprime:t2:n2048:seed3   n_zero_gates=  4069  beta=1.509306 qk=1.853493 g=1.086661  frac=0.4967041015625
arm_smprime:t2:n2048:seed4   n_zero_gates=  4123  beta=0.587580 qk=1.396958 g=1.200055  frac=0.5032958984375
arm_smprime:t2:n2048:seed5   n_zero_gates=  4123  beta=0.782138 qk=1.262158 g=1.123946  frac=0.5032958984375
arm_smprime:t2:n2048:seed6   n_zero_gates=  4123  beta=0.834885 qk=1.169418 g=1.301958  frac=0.5032958984375
arm_smprime:t2:n2048:seed7   n_zero_gates=  4123  beta=0.901199 qk=1.645161 g=1.450484  frac=0.5032958984375
```

Six of eight independently seeded, independently trained cells return the same
thirteen digits. The seventh returns a value whose sum with the first is exactly
`1.0`. The trained switches beside them (`beta` spanning `0.588–1.509`, `qk`
spanning `0.501–1.853`) are all different, so these are eight different models.

### The corpus, counted with no arm in the room `[RUN]`

Built exactly as `scripts/v15_r1.py:697-701` builds it — `head = S-1-T_STAR =
61`, `live = [62, 63]`, eval batch at the hardcoded `seed=12345`, `n_eval=4096`:

```
live band = [62, 63]   head = 61   n_eval = 4096   |a_live| = 8192
  a == -1 : 4123   fraction = 0.5032958984375
  a == +1 : 4069   fraction = 0.4967041015625
  a ==  0 : 0      fraction = 0.0
```

`4123` and `4069` are the corpus's own Rademacher sign counts. They are the
journal's two published values, to the last bit, and `n_zero_gates = 4123` is
the count itself.

### (ii) The runnable test and its verbatim RED

`tests/mars_v20/test_frac_gate_annihilated_is_the_corpus.py`, run at HEAD
`207e7b9`:

```
$ PYTHONPATH="..." python -m pytest tests/mars_v20/test_frac_gate_annihilated_is_the_corpus.py -v

test_control_the_census_is_not_degenerate_and_the_journal_is_readable PASSED
test_control_a_column_that_is_genuinely_the_arms_does_not_match_the_census PASSED
test_frac_gate_annihilated_is_not_the_corpus_sign_census FAILED
test_arm_pl_frac_gate_annihilated_can_take_a_second_value FAILED
test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus PASSED

E  AssertionError: 7/8 trained cells publish a gate diagnostic equal to a
   CORPUS-ONLY statistic. corpus: neg=4123 (0.5032958984375), pos=4069
   (0.4967041015625) of n=8192. cells:
   [('arm_smprime:t2:n2048:seed0', 0.5032958984375),
    ('arm_smprime:t2:n2048:seed1', 0.5032958984375),
    ('arm_smprime:t2:n2048:seed3', 0.4967041015625),
    ('arm_smprime:t2:n2048:seed4', 0.5032958984375),
    ('arm_smprime:t2:n2048:seed5', 0.5032958984375),
    ('arm_smprime:t2:n2048:seed6', 0.5032958984375),
    ('arm_smprime:t2:n2048:seed7', 0.5032958984375)]

E  AssertionError: all 8 ARM PL cells publish frac_gate_annihilated = {0.0};
   the column has zero variance across the arm's own tournament

2 failed, 3 passed in 1.97s
```

**Both controls are GREEN and they are load-bearing.** The first proves the
census is a real split (`n = 8192`, `0.4 < frac_neg < 0.6`, the two values
distinct) and the journal parses. The second runs the identical comparison
against `eval_nrmse` — a column that genuinely is the arm's — and finds no
coincidence, so the test is not a machine that condemns every column alike.

### (iii) The mechanism

**Two mechanisms in one column, and they are different.**

**ARM S-M-prime — a NEW instance of `MISTAKES.md` M-18 (`MISTAKES.md:1483`),
reached by the route M-18's own correction paragraph named and nobody re-ran the
check for.** M-18's catalogued instance is the `log|a|` linear probe, dead
because its regressand is constant (`SST = 0`). This column is not structurally
constant — an arm may annihilate any fraction in `[0,1]` — yet every trained cell
but one lands on the corpus's sign census. The route is the one M-18's correction
already established and this column never inherited: `scale/negation_scope.py`
assigns `x[:, :, CH_DRIVE] = a` at `:245` and `:432`, so **the gate is a visible
input channel**. The magnitude head has copied the sign of that channel — `m = 0`
exactly where `a_j = -1`, or the complement at seed 3 — and the diagnostic reads
the copy back as a learned property. The satisfied form of M-18's own registered
CHECK is the test above:

> *"A diagnostic whose no-arm value equals its expected with-arm value is
> measuring the corpus."* — `MISTAKES.md` M-18

**ARM PL — `V-10` / `M-5`, and the constant is a theorem.**
`scripts/v15_r1.py:361` reads the column off `lg = model.heads(x)[0]`, which is a
plain `nn.Linear` output and is finite at every parameter value, so
`~isfinite(lg)` is empty always. All eight cells publish `0.0` and no parameter
setting can move it. The column is a per-seed restatement of
`no_prefix_scan_represents_a_zero_gate` (`lean/CEQ/V16Domain.lean`), printed
eight times in the deciding journal as if it were eight measurements. **The two
arms' `frac_gate_annihilated` values sit in adjacent rows of the same table and
invite the reading "ARM PL annihilates 0%, ARM S-M-prime annihilates 50%" — a
comparison in which one side is a theorem and the other is the draw.**

### (iv) Replacement route — REPRICE

**What the column was FOR (the goal, not the method).** To answer: *does the
arm's gate use the annihilating endpoint the path-product construction exists to
provide?* That is the question the closed cap and `V16Domain.lean` clause 4 were
built to make askable, and it is a good question.

**The sharpest surviving route, named concretely enough to dispatch.**
`MISTAKES.md` M-18's own amendment already states it and it was written for
exactly this corpus: *"probe `sign(a_i)` off the arm's gate and report accuracy
`p`, which is discriminating where `log|a|` is not."* Concretely — the object is
`ceq/arm_smprime.ArmSMPrime.zero_hop_mask` (already shipped,
`ceq/arm_smprime.py:559`), the measurement is `ceq/arm_smprime.annihilation_mcc`
(already shipped, `ceq/arm_smprime.py:416`) evaluated with
`pred_zero = zero_hop_mask(x_ev)` against
`true_zero = (x_ev[..., CH_DRIVE] == 0)`, and the column that ships is the MCC,
not the marginal. **Neither object needs to be written; both are in the module
and neither is called by any producer of a deciding cell** —
`annihilation_mcc` appears in `ceq/arm_smprime.py` and in `tests/`, and nowhere
in `scripts/v15_r1.py`.

**Why it is sharper in one line.** A marginal rate cannot distinguish an arm that
annihilates the right positions from one that annihilates that many positions,
and MCC is exactly the statistic that can — it is zero for a gate that has copied
the marginal and nothing else.

**Second half of the route, and it is cheap.** The `frac_gate_annihilated` column
is not deleted; it is REPRICED into a control by printing the corpus's own
`frac_neg` beside it in the same record. The number that made this strike costs
one `int((a < 0).sum())` on a batch the runner already draws at
`scripts/v15_r1.py:701`.

---

## STRIKE 2 — `arm_phase`, `arm_pl` and `arm_smprime` are ONE operator at the corner, and the `exp_scan` planted negative separates only on the zero set

### (i) The claim struck, and where it lives

Four operator modules ship as four objects: `ceq/arm_pl.py`, `ceq/arm_phase.py`,
`ceq/arm_smprime.py`, plus the softmax control. `arm_phase` carries its own 31 KB
document (`V15_ARM_PHASE.md`), its own suite (`tests/arm_phase/`), and its own
certification rows at `V16_R1_DEVICE_READY.md:414` and `V16_ARM_SMPRIME.md:580`.
Inside `arm_smprime`, `ROUTES = ("product", "exp_scan")`
(`ceq/arm_smprime.py:89`) presents two constructions, one of them the shipped
PLANTED NEGATIVE for BIND 1 — *"kept in the SHIPPED module so BIND 1's rejection
region is occupied by real code"* (`ceq/arm_smprime.py:86-88`).

### The algebra `[DERIVED]`, then the measurement

At `beta = qk = g = 1` the S-M-prime quotient telescopes. With
`R = cumsum(log m)` and `Phi = cumsum(theta)`:

```
num_ij / Z_i = [ e^{R_i-R_j} e^{i(Phi_i-Phi_j)} e^{w_ij} ]
             / [ e^{R_i} sum_j' e^{w_ij' - R_j'} ]
             = e^{w_ij - R_j} / sum_j' e^{w_ij' - R_j'}  *  e^{i(Phi_i-Phi_j)}
```

`exp(R_i)` cancels. The right-hand side is `arm_phase.operator` at `s = 0`
verbatim, and with `theta = 0` it is `arm_pl.operator` at `(g = log m, s = 0)`.

### (ii) The runnable test and its verbatim RED

`tests/mars_v20/test_the_three_arms_are_one_operator_at_beta_one.py`, float64,
**two shapes**, strictly-positive magnitudes so both routes are defined:

| comparison | s=8, d=4, seed=15 | s=32, d=6, seed=3 |
|---|---|---|
| `arm_phase(s=0)` vs `arm_smprime(b=1, product)` | **2.775558e-16** | **8.106339e-16** |
| `arm_phase(s=0)` vs `arm_smprime(b=1, exp_scan)` | **1.118863e-16** | **4.996004e-16** |
| `arm_pl(g=log m, s=0)` vs `arm_smprime(th=0, b=1)` | **1.110223e-16** | **5.551115e-16** |
| `arm_smprime` product vs `exp_scan` (PLANTED NEGATIVE) | **3.140185e-16** | **7.550333e-16** |
| **CONTROL** `arm_smprime` `beta=1.0` vs `beta=0.9` | 1.113528e-01 | 2.369757e-01 |
| **CONTROL** `arm_smprime` `qk=1.0` vs `qk=0.5` | 2.330637e-01 | 2.240028e-01 |

**TOLERANCE, and why.** `1e-12`. That is `105x` the tree's own worst float64
identity-bind residual on this same journal — `9.547918011776346e-15`, the
`arm_pl` identity row of `results/v17k_r4_retake.jsonl` — so the bar cannot be
accused of being tuned to admit a reading. It was NOT read off `9.522e-03`: that
figure (`V17_R4_RETAKE.md`, ruling row 4) is a cross-device `eval_nrmse` delta
and is in the wrong units for an operator comparison. The floor used is the one
the tree measures in operator units.

**The separation is fifteen orders of magnitude.** A real dial move reads `1e-1`
at the identical shapes and seeds; the four claimed-distinct pairs read `1e-16`.

```
$ python -m pytest tests/mars_v20/test_the_three_arms_are_one_operator_at_beta_one.py -v

test_control_a_genuine_dial_move_is_visible_at_these_shapes PASSED
test_arm_phase_is_arm_smprime_at_the_beta_one_corner FAILED
test_arm_pl_is_arm_smprime_at_the_beta_one_phase_free_corner FAILED
test_the_two_smprime_routes_are_two_primitives_off_the_zero_set FAILED
test_the_exp_scan_planted_negative_has_a_nonempty_region_on_every_deciding_cell FAILED

E  AssertionError: arm_phase(s=0) and arm_smprime(beta=1,route=product) agree to
   2.775558e-16 at s=8,d=4,seed=15 -- below TOL=1e-12, which is itself 105x the
   tree's worst float64 identity residual. One operator, two names.

E  AssertionError: arm_pl(g=log m, s=0) and arm_smprime(theta=0, beta=1) agree
   to 1.110223e-16 at s=8,d=4,seed=15 -- below TOL=1e-12. One operator, two names.

E  AssertionError: the shipped route and its planted negative agree to
   3.140185e-16 at s=8,d=4,seed=15; they separate only where some m_k == 0

E  AssertionError: 1/8 deciding cells carry NO annihilating gate
   {'arm_smprime:t2:n2048:seed2': 0}; the two routes agree to 3.14e-16 off the
   zero set, so on these cells the planted negative cannot fire
```

### (iii) The mechanism

**Two findings, and the second is the one with teeth.**

**(a) `arm_phase` is `arm_smprime` with `beta` frozen at 1 — a RENAMING,
bounded.** The collapse is not a coincidence at the identity point `m = 1`; it
holds at every `m in (0,1]` and every `theta`. `arm_phase`'s entire operator
family at `s = 0` **is** `arm_smprime`'s at `beta = qk = g = 1`. What `arm_phase`
owns beyond it is one key-bias head; what `arm_smprime` owns beyond `arm_phase`
is the `beta` dial. The two modules are one primitive with the dials split
between them, and they were priced, certified and documented as two objects.
**Bounded honestly: `arm_phase` is NOT in the v17-K wing list** —
`results/v17k_r4_retake.jsonl` header reads
`"arms": ["arm_pl", "arm_smprime", "softmax"]` — so this does not inflate the
tournament. It inflates the certification surface and the cost table.

**(b) `MISTAKES.md` V-24 (`MISTAKES.md:1658`), a live instance on a deciding
cell.** BIND 1's rejection region for `exp_scan` is not "the prefix-scan
construction is wrong". Measured, it is exactly **"the prefix scan differs from
the path product on windows containing an exactly-zero magnitude, and nowhere
else"** — off that set the two routes agree to `3.14e-16` at both shapes. The
deciding journal then reports `n_zero_gates = 0` at
`arm_smprime:t2:n2048:seed2`. **On that cell the forbidden route and the shipped
route are the same map, so the planted negative's rejection region there is
empty.** `scripts/v15_r1.py:857`'s comment — *"`exp_scan` is reachable only
through the module's planted negative and never from a trained cell"* — is true
and is not the point: the point is that at seed 2 there is nothing to be
reachable, because the two routes coincide.

This is V-24's own closing "residual caution" applied to a bind that has not had
it applied: *"State the bind's information content in the words that are true of
it."*

### (iv) Replacement route

**RETIRE (a).** What `arm_phase` was FOR: exhibiting that a magnitude cap and a
phase twist can be carried through a causal softmax without leaving the softmax
class. That goal is **already met by `arm_smprime` at `beta = 1`**, which is the
same operator and additionally reaches the annihilating endpoint the phase arm's
`exp(C_i - C_j)` provably cannot (`no_prefix_scan_represents_a_zero_gate`).
`arm_phase` retires from the certification and cost surface and stays in the tree
as what it measurably is — the `beta = 1` slice, plus a key-bias head that
`arm_smprime` does not have. Concretely dispatchable: the rows to strike are
`V16_R1_DEVICE_READY.md:414`, `V16_ARM_SMPRIME.md:580` and the `arm_phase` lines
in `COSTS.md`, each of which prices a separate object. Sharper in one line: a
certification surface that prices one operator twice spends real GPU minutes
proving the same thing about the same map.

**REPRICE (b).** What the `exp_scan` negative was FOR: showing BIND 1 can fail.
It can — but only on the zero set, so its information content is the SIZE of that
set, and that number is already in the record. The route is to print
`n_zero_gates` as the negative's **power**, in the same row as the bind's
verdict, and to read a bind on a cell with `n_zero_gates = 0` as NO VERDICT
rather than PASS. Object: `scripts/v15_r1.py::fires` (`:433`), which today
returns `not (residual <= 0.1)` and has no arm for "the negative could not have
fired". Sharper in one line: a bind that passes where its own negative is inert
is reporting the draw, which is Strike 1's mechanism reappearing one level up.

---

## A DEFECT IN AN EXISTING STRIKE'S EVIDENCE COMMAND — filed, not struck

`STRUCK.md`'s entry for `0.743864` is the sharpest precedent in the repository
and its verdict is **unchanged**. Its stated evidence command, however, no longer
reproduces as written. `[RUN]`

```
$ git log -S"absorbing_boundary_kernel" --all --oneline | wc -l
9
$ git log -S"def absorbing_boundary_kernel" --all --oneline | wc -l
5
```

All five hits are `PREREGISTRATION_HOLE_AUDIT.md:421` and
`tests/cameron/test_harmonic_attribution.py:13` — the audit **quoting its own
command** — plus the `kaggle/snapshot/repo/` copies of both. Writing the absence
proof into the tree made the absence proof return hits.

**The repaired form, verified against a control that must return non-zero:**

```
$ git log -S"def absorbing_boundary_kernel(" --all --oneline -- "*.py" ":!tests/" | wc -l
0
$ git log -S"def path_product("             --all --oneline -- "*.py" ":!tests/" | wc -l
5
```

**NEW CLASS, proposed for `MISTAKES.md`: P-12, an absence proof falsified by the
act of recording it.** The mirror of `V-7` (*a search structurally incapable of
finding anything, read as absence*): here the search was capable and correct, and
publishing it destroyed its own reproducibility. The check is one line — an
absence command written into a document ships with the path exclusion that keeps
the document out of its own result, and ships a control symbol that the same
invocation must still find.

---

## ATTACKS THAT DID NOT FIRE

An adversary who reports only hits is not calibrated. Four attempted, four
failed, and the failures are informative.

**1. "`arm_pl` and `arm_smprime` are one wing in the v17-K tournament."**
REFUTED by the journal. The operator collapse (Strike 2) holds at
`beta = qk = g = 1`, and the deciding cells are **not** at that corner: the
trained switches read `beta` `0.733 / 0.897 / 1.344 / 1.509 / 0.588 / 0.782 /
0.835 / 0.901`, `qk` `0.501 – 1.853`, `g` `1.080 – 1.450`
(`results/v17k_r4_retake.jsonl`, `manifest.smp_values`). Training leaves the
corner on every seed. The wing list is two operators plus a control, as claimed.
This also disposes of a live worry about `RULING 10'`: a `beta` that pinned at 1
would have collapsed the tournament to one wing, and the measurement says it does
not.

**2. "The `SMP_FIELDS` identity block cannot tell two arms apart."** REFUTED.
`manifest.smp_values` carries `route`, `beta`, `qk`, `g`, `m_setting`,
`theta_setting`, `v_setting`, `m_max`, `v_max`, `n_zero_gates`, all populated from
the trained module. The test
`test_the_manifest_identity_field_n_zero_gates_carries_more_than_the_corpus`
PASSES: seed 2's `n_zero_gates = 0` is not a corpus count, so the field is not
wholly corpus-determined. The weakened true statement — 7 of its 8 observed
values are corpus counts — is Strike 1 and is filed there, not here.

**3. "The `exp_scan` planted negative is inert in the arm's own suite."**
REFUTED. `tests/arm_smprime/test_arm_smprime.py:126` and `:213` both run it on
`smp.bedm_draw`, which zeroes `m[1:1 + max(1, s//3)]` by construction
(`ceq/arm_smprime.py:302`), so the negative fires as `nan` there every time. The
empty-region instance is on a **trained deciding cell**, not in the suite, and
that is where Strike 2(b) filed it.

**4. "A chat-memory constant in the `0.743864` class exists on the v17-K
frontier."** NOT FOUND. Every path cited in `V17K_RULINGS.md` that this node
resolved exists on disk, and the round's unmeasured quantities are declared as
named `⟨SLOT ...⟩` markers with an owed-by node — `FLOOR_TRAIN_ABS_DLOSS`,
`FLOOR_CHUNK_SPEC`, `DECIDING_CELL_IDS`, `DECIDING_CELL_BITWISE_RETAKE`,
`DELTA_BETA_PER_PARAM`, `GRADIENT_CENSUS` — which is the opposite of the
fabrication pattern. **On the evidence this node gathered, the v17-K ruling
ledger does not carry a `0.743864`-class entry.** The forgery pressure this round
is not in the constants; it is in the columns, which is where both strikes
landed.

---

## WHAT THIS NODE COULD NOT VALIDATE

1. **The mechanism behind Strike 1 is inferred, not measured directly.** What is
   MEASURED is that `frac_gate_annihilated` equals the corpus sign census on 7 of
   8 cells. That `m = 0` lands *specifically* on the negative positions — i.e.
   that the head copied `sign(CH_DRIVE)` — is the natural explanation and is
   **not** established here, because the trained weights are not in the tree and
   this node ran no training (L-LEAN). The route in Strike 1(iv)
   (`annihilation_mcc` on `zero_hop_mask`) is exactly the measurement that would
   settle it, and it requires a run.

2. **Seed 2 is unexplained.** `arm_smprime:t2:n2048:seed2` reports
   `frac_gate_annihilated = 0.0`, `n_zero_gates = 0`, `a_hat_min = 0.340760`,
   `a_hat_max = 0.540044`, and the best `eval_nrmse` in the whole tournament
   (`0.203920` against a `floor_1` of `0.707107`). It is the one cell that escapes
   Strike 1 and the one cell that carries Strike 2(b). Whether those two facts are
   the same fact is not determined here.

3. **Nothing was run on CUDA.** Every number above is float64 on this box's CPU.
   The operator collapse is an algebraic identity and is not expected to be
   device-sensitive, but it was not re-taken on the certified 4060, and
   `V16_ARM_SMPRIME.md:580` records that `arm_phase.operator` **RAISES** under
   `use_deterministic_algorithms(True)` on cuda — so the `arm_phase` half of
   Strike 2 cannot be re-taken under the round's strict-forward regime at all.

4. **The `arm_pl` gate reading was not chased.** `a_hat_max` reads `12.77`,
   `49.66` and `116.01` at seeds 2, 3 and 7 — an unbounded gate on an arm whose
   three worst `eval_nrmse` cells are exactly those three seeds
   (`1.152280`, `1.113339`, `1.148927`). That is a live thread and this node did
   not pull it.

5. **No nurse-level breadth.** The subagent pool was refused (see head of this
   file). A single node grepping serially covered the four arm modules, the retake
   journal and the ruling ledger; it did **not** sweep the other ~60 result files
   or the 129 KB `AUDIT.md`. Absence of a third strike is absence of search, not
   evidence of absence — `MISTAKES.md` V-7, applied to this report.
