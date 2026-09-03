# V20 R15 it.2 — SATURN (WATSON)

Instruments, dossiers, the journal. Three tasks: the wing-distinctness
measurement that blocks the it.4 freeze, K5 bound or dropped, and the producer
search repaired after two of four K6 instances were struck.

New node: `tests/saturn/test_v20_r15_wings_distinct.py`.
Repaired node: `tests/saturn/test_v20_r15_wing_rubric.py`.

---

## A — WINGS-DISTINCT at the trained settings

### A.1 What `assert_arms_distinct` actually covers

`tests/loop/test_arms_distinct.py:69` [READ]. It fingerprints an arm by the
tuple of `ceq/bench.py::sign_flip_rate` readings across six seeds and raises if
two arms agree on all six. Its arm list is `ARMS` at
`tests/loop/test_arms_distinct.py:56` — `signed, softmax, sgate, paraformer,
signmag, deltanet, tgate, tgatex`. Seven bench arms plus softmax. **W1/W2/W3 are
not in it and cannot be: they are not `sign_flip_rate` kinds.** The it.1 flag
was correct — the three wings had never been measured against each other.

### A.2 The three wings are three different call signatures

[READ]

| Wing | Entry point | Signature | Reduction |
|---|---|---|---|
| W1 | `ceq/arm_smprime.py:234` `operator` | `(q,k,u,theta,*,beta,qk,g,route)` | `cumprod` path product / `Z^beta` |
| W2 | `ceq/arm_phase.py:158` `operator` | `(q,k,u,theta,s,*,diagonal)` | `cumsum(log m)` key bias × unimodular twist |
| W3 | `ceq/arm_pl.py:99` `operator` | `(q,k,g,s,*,diagonal)` | `cumsum(g)` key bias, REAL |

Comparing them needs a common gate content. The mapping used is the one
`ceq/compat.py:393-394` names: `arm_phase.key_bias` is `s - cumsum(log m)` and
`arm_pl.key_bias` is `s - cumsum(g)`, so **W3 is put on W2's gate by
`g = log(clamp(u,0,1))`**. Both modules define `magnitude` as exactly
`torch.clamp(u, 0.0, 1.0)` (`ceq/arm_smprime.py:109`, `ceq/arm_phase.py:85`), so
no re-parameterisation is smuggled in.

### A.3 Trained settings, READ not chosen

`results/v17k_r4_retake.jsonl`, `manifest.smp_values` on each `arm_smprime`
cell [READ]:

```
seed 0: beta 0.7325604557991028  qk 1.276558756828308  g 1.319505214691162  route "product"
```

eight such rows, `beta` ∈ {0.5875797867774963, 0.7325604557991028,
0.7821376323699951, 0.8348854184150696, 0.8969751000404358, 0.9011988043785095,
1.3439332246780396, 1.5093060731887817} — MARS's list, at full precision, and
**never 1.0**. Gate magnitude range from the same journal's `arm_pl` cells,
`a_hat_min`/`a_hat_max`: `[0.0318116, 116.006]`. The test reads both at run
time; nothing is typed in.

`m_setting` and `theta_setting` in the same block read `"learned: m_head(x)"`
and `"learned: theta_head(x)"` — **the trained gate vectors themselves are not
journalled**, only their range. That is the limit A.6 rests on.

### A.4 Tolerance

`1e-12`. That is 105× `9.547918011776346e-15`, the worst float64 identity
residual this tree records for itself (`results/v17k_r4_retake.jsonl`, `arm_pl`
bind at `s=64`, `"residual"`). MARS's tolerance, reused rather than
re-invented; the test asserts the residual it is scaled from is still in the
journal. It is **not** `9.522e-03` — that is a cross-device `eval_nrmse` delta
and the wrong units for an operator comparison.

### A.5 RED first, verbatim

The bind's own assertion, run with MARS's corner substituted for the trained
settings [RUN]:

```
AssertionError: at s=8 these wing pairs are the SAME operator to 1e-12:
{'W1/W2': 1.1102230246251565e-16, 'W1/W3': 1.1102230246251565e-16, 'W2/W3': 0.0}
s=8 corner readings: {'W1/W2': '1.110223e-16', 'W1/W3': '1.110223e-16', 'W2/W3': '0.000000e+00'}
```

This reproduces MARS's corner reading independently (`1.11e-16`) and adds one he
did not report: **W2 and W3 read exactly `0.0` at the corner — bitwise
identical, not merely close.** The instrument can detect sameness. Two planted
positives are standing tests: an arm against itself must read exact `0.0`
(`test_the_measure_calls_an_arm_against_itself_the_same`) and the corner must be
called SAME (`test_the_measure_calls_marss_corner_the_same`).

### A.6 GREEN, and the measurement

[RUN] `python -m pytest tests/saturn/test_v20_r15_wings_distinct.py -q -s`,
5 passed. Reported figure is **min over the eight trained seeds** of
`max |A − B|` over the operator matrix — the weakest separation, not the best:

| shape | W1/W2 | W1/W3 | W2/W3 |
|---|---|---|---|
| `s=8`  | `3.063578e-01` | `6.702194e-01` | `6.907369e-01` |
| `s=64` | `3.389808e-01` | `6.657651e-01` | `9.642279e-01` |

Every pair separates by ≥ `0.30`, i.e. **11 orders of magnitude above the
tolerance**, at both shapes, on all eight trained `(beta, qk, g)` triples.

### A.7 Verdict on N — and what the measurement cannot settle

**W1 is distinct from W2 and from W3 at the journalled trained settings.** That
is measured, at two shapes, with a control that separates. N ≥ 2.

**W2 vs W3 is NOT settled, and I will not claim it is.** Two reasons, both
measured:

1. `ceq/arm_phase.py:158` returns `p * phase_factor(theta)` where `p` is
   exactly `ceq/arm_pl.py:99`'s matrix under `g = log m`. So `|W2| = W3`
   identically and **the entire W2/W3 separation is carried by `theta`**. At
   `theta = 0` they are bitwise equal — the `0.000000e+00` in A.5. The
   `6.9e-01`/`9.6e-01` above come from a `theta` **drawn** uniformly on
   `(−π, π]`, not read from any record.
2. `theta` cannot be read: `grep -c '"kind": "arm_phase"' results/*` is **0**
   [RUN] — `arm_phase` has **no trained cell anywhere in `results/`**. W2 has no
   trained setting to be measured at.

So: **N = 3 if W2's `theta_head` trains away from zero, N = 2 otherwise, and the
record does not say.** The it.4 freeze should be gated on one W2 training cell
with `theta` journalled, not on this file. Everything else the freeze needed
from this measurement is closed.

---

## B — K5: WITHDRAWN, refuted by its own node

K5 as filed: "GPU-seconds-to-floor is unmeasurable for every wing." The node was
written in the shape the Inspector proposed — criterion (3) has a computable
value iff some cell crosses the floor — and it **refuted the claim** [RUN]:

```
AssertionError: arms with a computable seconds-to-floor:
{'arm_pl': 1.884, 'arm_smprime': 47.048}
```

Measured from `results/v17k_r4_retake.jsonl`: `floor_1 = 0.7071067811865476`,
24 cells, **6 cross** — `arm_pl` at seeds 0/1/4/5/6 (`eval_nrmse` 0.6447,
0.6445, 0.6337, 0.6420, 0.6621) and `arm_smprime` at seed 2 (0.2039). The
journal's own `dist_to_floor` is negative on each, which is its sign convention
for below-floor. Criterion (3) therefore has a value and it ranks: **W3 1.884
GPU-s vs W1 47.048 GPU-s to first crossing, 25.0×**.

The it.1 citations were not false about their own subjects — `V15_R1.md:250`
`crosses? NO` is the v15 **CPU `arm_pl`** round and `V17_R4_RETAKE.md:194` is
about **sign-flips**, not seconds-to-floor. Generalising them to "every wing"
was the defect: neither is a statement about the retake journal's `eval_nrmse`
against `floor_1`.

**Replacement route (reroute, not retire).** What survives is strictly narrower
and is now the standing green:
`test_criterion_three_is_computable_for_w1_and_w3_and_absent_for_w2` —
criterion (3) is computable for W1 and W3, ranks W3 ahead of W1, and is
**undefined for W2 for want of a run, not for want of a crossing**. The test
fails the moment `arm_phase` gains a cell, which is the same event A.7 needs.

---

## C — K6: producer search repaired, re-run, and the mechanism

### C.1 The defect

`producers_for` (`tests/saturn/test_v20_r15_wing_rubric.py`) is a literal
substring search, and its only calibration control,
`test_the_producer_search_finds_a_number_that_does_have_one`, is the **number**
`0.7071067811865476`. It certifies the search for compact literals. Iteration 1
registered four tokens, **two of them English phrases** — `"d=20 at the same"`
(annex prose) and `"0.492 vs KL 0.519"`. No `.py` emits an English phrase, so
their zero-hit readings were **structural, not evidential**. Wilson §7 M13 names
live producers for the second — `scale/foreman_consequence.py:12`,
`scale/foreman_signfloor.py:382` — and neither is under `attic` or `kaggle`, so
neither was excluded by this file's own filter. Both were struck, correctly.

### C.2 The repair

`ADMISSIBLE_TOKEN = re.compile(r"^\S*\d\S*$")`: a token carrying a digit and no
whitespace. `producers_for` now **raises `ValueError` at registration** rather
than returning `[]`. Calibration, all green [RUN]:

* `test_the_producer_search_rejects_a_phrase_instead_of_reading_zero_hits` —
  `"d=20 at the same"` must RAISE (RED-first for the repair);
* `test_the_producer_search_rejects_the_other_struck_token_too`;
* `test_every_registered_token_is_inside_the_certified_domain` — the
  registration gate;
* `test_the_calibration_control_is_itself_inside_the_domain` — the rule covers
  the control, or it is a rule for other people;
* `test_the_withdrawn_tokens_are_recorded_rather_than_deleted` — the two struck
  tokens sit in `UNBINDABLE_BY_SUBSTRING` with their reason, withdrawn in
  writing, not quietly dropped.

### C.3 K6 re-run: CONFIRMED on the two survivors

[RUN] `test_every_annex_run_instance_has_a_producer_in_the_tree` is the standing
RED and still fires on exactly the admissible pair:

```
AssertionError: annex items marked [RUN] whose figure NO .py/.json/.jsonl/.txt/.log
in this tree can emit: {'M9-F1  Cantelli/Boole cutoff': 'd=65',
'M2     gate-landscape theta': '1.3e-3'}
```

K6 stands at **2 instances, not 4**. Cost of the repair, stated: two of the four
filed instances are withdrawn, and the claim's reach halves.

### C.4 For `MISTAKES.md` — the entry

> **V-7b — the rule enforced on one half of your own instrument.**
> `tests/saturn/test_v20_r15_wing_rubric.py` cites V-7 by name in its own module
> docstring ("a search structurally incapable of finding anything, read as
> absence") and binds it on the clause-(c) **absence** search, with a control —
> a kill known to be in the record at `ARSENAL.md:303` that the search must
> find. The same file's **producer** search got no such gate: its calibration
> control was a number, its registered tokens included English phrases, and
> nothing checked that the tokens lay inside the domain the control certifies.
> Two of four K6 instances were struck for it.
> **Mechanism:** a rule is written against a named past failure, then applied to
> the instrument half that failure occurred in, while the other half of the same
> file inherits the citation without the check. The docstring citation reads as
> coverage. **Detection:** for every rule a test file names, list every search in
> that file and show the rule is bound on each. **Repair shape:** reject
> out-of-domain input at REGISTRATION, and assert the calibration control itself
> satisfies the registration rule.

---

## Suite state

[RUN] `python -m pytest tests/saturn/ -q` — 59 passed, 7 failed.
New/changed here: 5 passed in `test_v20_r15_wings_distinct.py`; in
`test_v20_r15_wing_rubric.py` the K6 RED (`..._has_a_producer_in_the_tree`)
fires as designed and everything else is green. The other 6 failures are
pre-existing standing REDs from earlier rounds
(`test_journal_path_is_discoverable.py` ×3, `test_r10_it2_spotcheck_reds.py` ×3)
and are untouched by this iteration.

## Limits

The W2/W3 half of the distinctness measurement is conditional on a drawn
`theta`, because no `arm_phase` cell exists in `results/`; N is 3 or 2 and this
file cannot say which. W1's trained gate vectors `m_head(x)`/`theta_head(x)` are
likewise unjournalled, so W1's readings use a gate drawn from the journalled
range `[0.0318116, 116.006]` rather than the trained vectors themselves — the
`(beta, qk, g)` switches are read, the vectors are not. `seconds_to_floor`
counts cumulative `secs` over journal order to the first crossing cell, which is
a per-seed training cost summed in file order, not a wall-clock schedule. No git
writes, nothing touched Kaggle.
