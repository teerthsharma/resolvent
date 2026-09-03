# V20 R15 it.3 — SATURN (WATSON)

MARS's STRIKE 5 is upheld. The it.2 headline "every pair separates by `>= 0.30`
at both shapes at the trained settings" is **VOID and is withdrawn**. It is
replaced by a measurement on the gate W1 can actually occupy, and the replacement
number is smaller. The distinctness verdict itself survives; the threshold does
not.

New node: `tests/saturn/test_v20_r15_wings_distinct_percell.py` — **1 RED,
12 GREEN**. The old node `tests/saturn/test_v20_r15_wings_distinct.py` keeps
running with its bind marked SUPERSEDED in the docstring, so the void gate stays
visible next to its replacement rather than being quietly deleted.

No git write. Nothing touched Kaggle. One moon of four.

---

## TASK A — STRIKE 5 UPHELD. THE `>= 0.30` DOES NOT SURVIVE.

### A.1 The reroute, executed as filed

MARS's reroute taken verbatim: `u` drawn per cell on **that cell's own**
`[a_hat_min, a_hat_max]`, with an atom at exactly `0.0` of mass
`n_zero_gates / 8192`. Every input `[READ]` from
`results/v17k_r4_retake.jsonl`, the `arm_smprime` cell rows — `a_hat_min` and
`a_hat_max` at top level, `n_zero_gates` in `manifest.smp_values`:

| seed | `a_hat_min` | `a_hat_max` | `n_zero_gates` | atom mass |
|---|---|---|---|---|
| 0 | 0.0 | 1.0 | 4123 | 0.5032958984375 |
| 1 | 0.0 | 1.0 | 4123 | 0.5032958984375 |
| 2 | 0.340760201215744 | 0.5400443077087402 | **0** | 0.0 |
| 3 | 0.0 | 0.8454325795173645 | 4069 | 0.4967041015625 |
| 4–7 | 0.0 | 1.0 | 4123 | 0.5032958984375 |

The denominator `8192` is not assumed. `test_the_atom_mass_is_the_journals_own_annihilated_fraction`
is GREEN: `n_zero_gates / 8192 == frac_gate_annihilated` **bitwise on all eight
cells**. The atom's mass is read from the record, not chosen.

### A.2 The control MARS demanded — one that fires if the gate were still saturated

`test_the_void_draw_saturates_the_cap_and_the_per_cell_draw_does_not` [RUN],
GREEN. Both halves in one test so the detector is seen to work before its
"clean" verdict is used:

* **planted positive** — the void draw `U(0.0318116, 116.006073)` under
  `ceq/arm_smprime.py:109`'s `clamp(u,0,1)` reads **99.109%** of the gate at
  exactly `1.0`. MARS's analytic `99.17%` is confirmed by measurement to three
  figures. Had this read low, nothing below could be trusted.
* **the corrected draw** — fraction at exactly `1.0` is **0.000000 on all eight
  cells**. `a_hat_max <= 1.0` per cell, so the clamp is never engaged.

The it.2 planted positives are carried forward unchanged: an arm against itself
reads exact `0.0`, and the strike's endpoint claim is bound —
`test_the_per_cell_support_reaches_the_annihilating_endpoint` asserts
`a_hat_min == 0.0` on seeds `[0,1,3,4,5,6,7]` and `VOID_LO > 0.0`.

### A.3 The measurement — min over the eight trained cells of `max |A - B|`

Continuous part of the draw (atom excluded; the atom is A.4, and it is not a
number). `[RUN]` `python -m pytest tests/saturn/test_v20_r15_wings_distinct_percell.py -q -s`:

| shape | pair | it.2, void gate | **it.3, per-cell gate** | change |
|---|---|---|---|---|
| `s=8`  | W1/W2 | `3.063578e-01` | **`2.623589e-01`** | **−14.4%, below 0.30** |
| `s=8`  | W1/W3 | `6.702194e-01` | `5.207431e-01` | −22.3% |
| `s=8`  | W2/W3 | `6.907369e-01` | `3.694346e-01` | −46.5% |
| `s=64` | W1/W2 | `3.389808e-01` | `4.202484e-01` | +24.0% |
| `s=64` | W1/W3 | `6.657651e-01` | `1.182163e+00` | +77.6% |
| `s=64` | W2/W3 | `9.642279e-01` | `1.317549e+00` | +36.6% |

### A.4 STATED PLAINLY

**The `>= 0.30` separation does NOT survive. It is withdrawn.** At `s=8` the
W1/W2 pair reads `2.623589e-01`, below the published floor, and the it.2
sentence "Every pair separates by `>= 0.30` ... at both shapes" is false on the
corrected gate. That is the RED, verbatim:

```
E  AssertionError: at s=8 the it.2 published floor 0.3 is not met on the
   per-cell trained gate: {'W1/W2': '2.623589e-01'}. The separation SHRINKS;
   it does not vanish -- every reading is still >= 1e11 x the tolerance 1e-12
E  assert not {'W1/W2': '2.623589e-01'}
```

**It shrinks; it does not vanish.** The distinctness verdict is unchanged:
the weakest of the six readings is `2.623589e-01`, which is `2.6e11 x` the
tolerance `1e-12`. `test_the_wings_still_separate_on_the_per_cell_trained_gate`
is GREEN at both shapes with no pair inside tolerance. **N >= 2 stands.**

Note also that the correction is **not a uniform shrink**: `s=8` falls on all
three pairs and `s=64` rises on all three. A gate defect that moved every number
one way would be a scale error; this one changes the shape-dependence, which is
what a saturated gate does — at `s=8` the void draw's 99% identity gate left the
`beta/qk/g` switches as almost the whole signal, and those switches act more
strongly at short rows.

### A.5 THE ATOM — a domain difference, and it is reported as one, not as a number

`test_at_the_annihilating_atom_w2_and_w3_are_undefined_where_w1_is_not` [RUN],
GREEN at both shapes. With the atom in the draw:

* **W1 returns a finite matrix on all 8 cells.** `cumprod` on a zero gate is a
  true `0`.
* **W2 and W3 both return `nan` on exactly the 7 cells that contain an exact
  zero gate.** Both route `log 0 = -inf` into a softmax logit
  (`ceq/arm_phase.py:126`, `ceq/arm_pl.py:97`), and `arm_phase.scan_phase`'s own
  docstring (`ceq/arm_phase.py:115-119`) says so in advance: "UNDEFINED PAST A
  ZERO GATE".
* The **one** cell where all three are defined at the atom is **seed 2**, the
  cell with `n_zero_gates == 0` — MARS's STRIKE 3 escape cell, arrived at here
  from the gate rather than from the manifest.

`nan` is returned **as** `nan` by this node's `delta` and never widened to
`inf`. An `inf` separation reported as a measurement would be an artifact
promoted to evidence; the definedness structure is the finding.

This is the same object as JUPITER's STRIKE-7 domain exclusion reached from a
fourth direction, and it sharpens it: the exclusion is not a measure-zero corner
on the trained record, it carries **50.3% of the gate mass** on seven of eight
cells.

---

## TASK B — THE W2↔W3 LEG, MEASURED

Nobody had run `pl <-> phase`. It is run here, on the corrected per-cell gate.

| shape | continuous draw | with the atom (seed 2 only; 7/8 undefined) |
|---|---|---|
| `s=8`  | `3.694346e-01` | `6.017639e-01` |
| `s=64` | `1.317549e+00` | `1.382628e+00` |

### B.1 The leg is `theta` and nothing else, and that is now bound, not argued

`test_the_w2_w3_leg_is_carried_entirely_by_theta` [RUN], GREEN at both shapes,
over all eight per-cell gates. `ceq/arm_phase.py:179` returns
`p * phase_factor(th)` where `p` is `ceq/arm_pl.py:99`'s matrix under
`g = log m`, so `|W2| = W3` identically. The test asserts, per cell:

* `delta(W2, W3) == 0.0` **exactly** when `theta` is zeroed on that same gate —
  bitwise, not within tolerance;
* `delta(W2, W3) > TOL` at the drawn `theta`.

So every digit in the table above is `theta` and no part of it is the gate.

### B.2 The `theta` used, and where it came from — IT IS DRAWN

**`theta ~ U(-pi, pi]`, one vector per cell, seeded by that cell's own seed. It
is DRAWN, not fit, and W2 has no trained cell to fit it from.** `[RUN]`
`test_the_w2_theta_is_drawn_and_not_trained_anywhere_in_results` sweeps every
file in `results/` for `"kind": "arm_phase"` and finds **none**; it is written
to flip RED the moment one lands, which is the re-measure trigger.

The consequence is unchanged from it.2 and is not softened by having a number:
**the W2↔W3 leg is a measurement of a drawn parameter.** It answers JUPITER's
`W2<->W3 UNMEASURED` in the sense that the leg is no longer unrun, and it does
not answer it in the sense that would let the wing list freeze at three.

`theta_head` exists and is trainable — `ceq/arm_phase.py:486`
`self.theta_head = nn.Linear(d_model, 1)`, zeroed to the identity by
`identity_heads` at `:492-499`. Nothing has ever trained it: the class docstring
at `ceq/arm_phase.py:476` reads "NOT TRAINED HERE and not by this node
(L-LEAN). No optimizer, no gradient."

---

## TASK C — THE W2 CELL: NOT RUN, AND THE STOPPING REASON IS NOT THE ONE EXPECTED

No cell was faked. No cell was started.

### C.1 The determinism objection is REFUTED for this round

`V16_ARM_SMPRIME.md:576-581` [READ] records the RAISE under **strict** mode:

```
  arm_smprime.operator        under use_deterministic_algorithms(True) on cuda: OK
  arm_phase.operator          under use_deterministic_algorithms(True) on cuda: RAISES
      RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
  arm_phase.scan_phase        under use_deterministic_algorithms(True) on cuda: RAISES
```

**This round's regime is not strict.** The deciding journal's own header,
`results/v17k_r4_retake.jsonl:1` [READ], declares
`"deterministic_algorithms": true, "deterministic_warn_only": true`, and its
producer sets exactly that: `scripts/v15_r1.py:575`
`torch.use_deterministic_algorithms(True, warn_only=True)`. The same file's
docstring (`:91-93`) states why — strict mode is not executable on a cell that
trains at all, because autograd differentiates `cumprod` with `cumsum`, so
**`arm_smprime`'s own backward would raise under strict too**. Under
`warn_only=True` `cumsum` warns and runs.

**So the round's determinism regime does not forbid an `arm_phase` cell.** The
premise that it might is withdrawn. It would forbid one under a regime this
round does not use, and no `arm_smprime` cell could be taken under that regime
either.

### C.2 The real blocker: the harness cannot construct an `ArmPhase`

`scripts/v15_r1.py` is the producer of the deciding journal (`:578`
`results/{tag}.jsonl`). [READ]

* `:146` `GATED_ARMS = ("arm_pl", "arm_smprime")`
* `:149` `ARM_MODULES = {"arm_pl": arm_pl, "arm_smprime": arm_smprime}`
* `:179-184` `make_arm` has branches for those two and falls through to
  `scale/m3_capability.py::Arm(kind, s)`
* `scale/m3_capability.py:104-105` `if kind not in ARMS: raise ValueError(kind)`

`arm_phase` is imported by `scripts/v15_r1.py:118` but is used only for
`arm_phase.winding(theta)` at `:404`. **`--arms arm_phase` raises `ValueError`
at model construction.** That is the correct behaviour and worth stating
positively: the harness **refuses** rather than silently journalling a
`kind: "arm_phase"` row produced by a bench arm. The cell this round's binding
kill exists to prevent is not reachable by accident.

### C.3 The price, stated rather than started

GPU time is **not** the cost. From the same journal's `secs` field, the eight
`arm_smprime` cells cost `14.852`–`16.660` s each at 150 steps / `n_train=2048`
on this round's CUDA. One `arm_phase` cell is that order — call it **~16 GPU-s**,
eight cells **~130 GPU-s**.

The cost is a **producer edit to the deciding journal, mid-round**: a `make_arm`
branch, an `ArmPhase` train/eval path (there is no optimizer path — `:476`), and
a `PHASE_FIELDS` manifest emit so `theta` is actually journalled
(`ceq/arm_phase.py:75-76` already declares the field tuple, so the manifest half
is ready). Under a 20-minute wall clock, with the strike answers in Tasks A and
B unfinished at the time the decision was taken, that edit was not attempted. It
is priced here so the next node starts from a number instead of from a guess.

### C.4 Replacement route for the it.4 freeze — two, both priced

1. **RETIRE, zero cost.** Freeze the wing list at **N = 2 measured (W1, W3)**
   and carry W2 as **UNPRODUCED**, not as a wing. This is honest against
   everything above: W2's magnitude is W3's identically, the whole separation is
   a drawn `theta`, and no run has ever produced one.
2. **REROUTE, ~130 GPU-s plus one harness iteration.** Budget one iteration to
   the three edits in C.3, then eight cells. The determinism gate does not need
   to move; C.1 removes it.

The standing RED that decides between them is already in the tree:
`test_the_w2_theta_is_drawn_and_not_trained_anywhere_in_results` goes RED on the
first `arm_phase` cell to land, and
`test_criterion_three_is_computable_for_w1_and_w3_and_absent_for_w2` (it.2, §B)
fails at the same event.

---

## WHAT CHANGED IN THE TREE

* **NEW** `tests/saturn/test_v20_r15_wings_distinct_percell.py` — 1 RED (the
  struck `0.30` floor, kept RED as the record of the strike), 12 GREEN.
* **EDITED** `tests/saturn/test_v20_r15_wings_distinct.py` — the bind gains a
  docstring marking it SUPERSEDED by the per-cell node, naming the void gate and
  the measured 99.1% saturation. The test still runs; its numbers are no longer
  the round's.

Suite: `[RUN]` `python -m pytest tests/saturn/ -q` — **71 passed, 8 failed**
(it.2 read 59 passed / 7 failed; +12 green and +1 red are this node's). The 7
carried failures are the 6 pre-existing standing REDs from earlier rounds
(`test_journal_path_is_discoverable.py` x3, `test_r10_it2_spotcheck_reds.py` x3)
plus the it.2 K6 RED, all untouched. The 8th is
`test_the_it2_published_floor_of_030_survives_the_corrected_gate[8]`, this
iteration's strike, RED by design.

---

## Limits

Every number in this file is float64 on this box's **CPU**; nothing was run on
CUDA this iteration, and the journalled cells being read were taken on CUDA. The
gate is drawn from the journalled per-cell **support and zero-mass**, which is
strictly more of the trained gate than it.2 used and is still not the trained
vectors — `m_setting` reads `"learned: m_head(x)"` and the vectors themselves are
not journalled, so a draw matching two moments of the true gate is the most the
record supports. Reported figures are the **min over the eight cells**, the
weakest separation, so a per-cell reading can be much larger and none is smaller.
`theta` on the W2 legs is drawn on `(-pi, pi]` and is not a trained value; W2 has
no trained cell. The atom leg's `s=8` and `s=64` draws hit the atom on 7 of 8
cells by construction of the Bernoulli, not by an exhaustive enumeration of the
zero pattern. `frac_gate_annihilated` is taken as the journal's own record of the
zero mass and is not independently recomputed from weights that are not in the
tree. No git write, nothing touched Kaggle.
