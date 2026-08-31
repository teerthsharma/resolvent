# MARS's four standing attacks — filed at it.0, bound by Saturn

Filed against `CEQ_V15_CONTRACT.md` PART IV's own list: "floor crossing by
feature leak (Neumann-term census on the reading tensors); parity by
underpowering (achieved-power column); the skyline leaks the oracle (gate
values never in `x`); guards chosen after seeing itineraries (journal
timestamp ordering)." Filed before R1's data exists, per the contract's own
rule that an attack filed after the numbers is a rationalisation.

Each attack is a runnable value test under `tests/mars_v15/`, each carries a
must-fire fixture that is shown failing on a planted positive, and each real
(non-must-fire) test either SKIPS with a named reason against a not-yet-built
artifact or runs live against a real artifact today. Full suite, this session,
`python -m pytest tests/mars_v15/ -v`: **21 passed, 4 skipped, 0 failed.** No
test passes on an absent subject (MISTAKES.md V-6/V-7): every SKIP names what
is missing and points at the must-fire proving the check is not vacuous.

## One line each

| # | attack | file | status |
|---|---|---|---|
| 1 | Floor crossing by feature leak | `tests/mars_v15/test_neumann_leak_reading_tensors.py` | **SKIP** (ARM PL not built) — 3/3 live tests GREEN (1 clean control, 2 must-fire) |
| 2 | Parity by underpowering | `tests/mars_v15/test_resolution_statement_achieved_power.py` | **SKIP** (no R4 row yet) — 10/10 live tests GREEN (instrument self-checks, MC cross-check, 2 must-fire) |
| 3 | The skyline leaks the oracle | `tests/mars_v15/test_skyline_gate_containment.py` | **GREEN today** (live census on the real BED-M-adjacent corpus reads clean) + **SKIP** for the not-yet-built v15 skyline — 4/4 other live tests GREEN |
| 4 | Guards chosen after itineraries | `tests/mars_v15/test_guard_itinerary_timestamp_order.py` | **SKIP** (BED-1 not built; 0 itinerary events in the 10,810-event journal) — 4/4 other live tests GREEN |

---

## Attack 1 — Floor crossing by feature leak

**Mechanism, one sentence.** ARM PL's gate `g(x_i) = -softplus(W x_i)` is
declared order-0 in the Neumann sense (`(I-A)^-1 = Σ A^t`; the label's t-hop
term is the path product `a_{s-1}...a_{s-t}.b`) — a function of `x_i` alone —
and if it instead depends on tokens up to `t*` positions away (a lookahead
tap, a cached neighbour feature), R1's `PL < floor_1 = 0.7071` would be
measuring a leaky reading pipeline, not the scan's compositional power.

**Class.** MISTAKES.md **M** (measurement failure). Adjacent to **M-1**
("train and eval saw different preprocessing" — here the reading step sees
more than its declared receptive field) and to **D-2** ("an oracle that is
the arm's own resolvent" — here the reading supplies what the resolvent was
supposed to earn). **New mechanism**: no existing entry names a
Jacobian-detectable receptive-field leak in a per-token reading tensor.

**File.** `tests/mars_v15/test_neumann_leak_reading_tensors.py`

**Status, actually run.**
```
test_census_is_silent_on_the_contract_specified_reading_function PASSED
test_census_MUST_FIRE_on_a_planted_order_2_leak PASSED
test_census_MUST_FIRE_at_every_leak_distance_it_is_asked_to_find PASSED
test_neumann_census_on_arm_pl SKIPPED
  reason: ARM PL not found (tried ceq.arm_pl / scale.arm_pl reading_gate /
  log_a). R1 has not run: V15_LEDGER.md NEXT=it.3-4, Lean train-gate verdict
  is it.5, ARM PL build is it.6-7, R1 is it.9-10.
```

**Must-fire, verbatim** (`python tests/mars_v15/test_neumann_leak_reading_tensors.py`):
```
demo OK: clean reading tensor cleared the census; planted order-2 leak found
at magnitude 0.087500 and the assertion raised on it
```
The census's own `assert_reading_tensor_clean` raised
`AssertionError: Neumann-term census found order(s) [2] present in a reading
tensor declared order<=0` against the planted fixture, and stayed silent on
the clean, spec-matching reading function (`census[0] > 1e-6` also checked,
so the census is shown capable of finding *something*, not merely quiet).

---

## Attack 2 — Parity by underpowering

**Mechanism, one sentence.** `CEQ_V15_CONTRACT.md` retires TOST below N=23
and substitutes "resolution statements" ("excludes a difference beyond
`Delta = t_{.975,N-1}.sd/sqrt(N)` and nothing smaller"), and a reader who
takes a small `Delta` as reassurance is reading a bound on what the design
*could* detect as a claim about what the arms *actually do* — at the round's
own registered N=8, achieved power against a plausible half-sigma true
difference is only ≈23%, so "no difference beyond Delta" is close to "this
design was not equipped to tell you either way."

**Class.** MISTAKES.md **M** (measurement failure). A **sharper instance of
M-13** ("an equivalence margin registered without a reachability check") and
**M-9** ("a verdict whose finest achievable p cannot reach the α it
quotes"). M-13 was filed against TOST; the contract's own fix for M-13 is to
retire TOST below N=23 and substitute resolution statements, and this attack
shows the substitution reopens the same disease unless every row carries the
achieved-power column M-13's own "Check" paragraph already prescribes.

**Independent confirmation already in the tree.** `V15_CONTRACT_ARITHMETIC_AUDIT.md`
§A-1, filed this round, computes achieved power at N=23 for the two-sample
TOST framing at **6.69%** and states: *"Mars's standing attack #2 is 'parity
by underpowering' and requires exactly that achieved-power column. The attack
and the clause it attacks are both in the same contract."* `V15_LEDGER.md`
NEXT item 4 instructs the coordinator to wire the two together; this test is
that wiring, in the one-sample framing PART I actually licenses (df=N-1, not
the two-sample df=2N-2 A-1 used).

**File.** `tests/mars_v15/test_resolution_statement_achieved_power.py`

**Status, actually run.**
```
test_achieved_power_at_zero_effect_is_alpha PASSED
test_achieved_power_matches_independent_monte_carlo[8-0.5]  PASSED
test_achieved_power_matches_independent_monte_carlo[23-0.5] PASSED
test_achieved_power_matches_independent_monte_carlo[70-0.5] PASSED
test_achieved_power_matches_independent_monte_carlo[23-1.0] PASSED
test_achieved_power_at_the_rounds_registered_n8_is_alarmingly_low PASSED
test_achieved_power_climbs_toward_070_as_n_grows_toward_the_tost_floor PASSED
test_require_achieved_power_column_MUST_FIRE_on_a_planted_row_without_one PASSED
test_require_achieved_power_column_MUST_FIRE_on_a_planted_wrong_number PASSED
test_require_achieved_power_column_passes_on_a_correctly_labelled_row PASSED
test_every_published_resolution_row_carries_a_correct_achieved_power_column SKIPPED
  reason: no resolution-statement-shaped row found under results/*.jsonl; R4
  has not run (V15_LEDGER.md NEXT=it.3-4, R4 scheduled it.15-17).
```
(The scan's keyword pattern was tightened during this filing after it
false-positived on `results/journal_census.jsonl`'s unrelated "the resolution
limit on any difference between two cells swept at different thread counts"
— a live V-4-shaped near-miss, corrected before it shipped, not left in.)

**Must-fire, verbatim** (`python tests/mars_v15/test_resolution_statement_achieved_power.py`):
```
demo OK: power(null)=0.0500==alpha, power(N=23,0.5sd)=0.6302 (mc 0.6311),
power(N=70,0.5sd)=0.9848, planted row without a power column correctly
raised MissingAchievedPowerError
```
A planted row shaped exactly like the contract's own example — N=8, a real
`sd`, `delta_res` printed, no `achieved_power` — raised
`MissingAchievedPowerError` naming the row's own implied power (0.232, well
under even odds); a second planted row carrying a copy-pasted
`achieved_power=0.80` that does not follow from its own (sd, n) also raised.
The formula was cross-checked against an independent from-scratch Monte
Carlo at four (N, effect) combinations, all agreeing within simulation
noise (≤0.03), before being trusted for the row-level rule.

---

## Attack 3 — The skyline leaks the oracle

**Mechanism, one sentence.** BED-M's skyline is an oracle-informed reference
allowed to know the true per-step gate `a` and compute the exact resolvent
from it, which is fine and by design — but if a **composed** (Neumann order
≥ 1) quantity such as a running partial resolvent or cumulative log-gate sum
also ends up recoverable from a channel of the tensor `x` that arms are
scored on, then "distance-to-native-skyline" stops meaning what the round
says it means, because an arm could read the skyline's own answer off its
input instead of composing anything.

**Not the strawman version.** `scale/negation_scope.py` (the live corpus
infrastructure the contract's "BED-M (Markov, exists)" refers to) puts the
**raw**, single-step gate `a_i` directly into `x` — documented explicitly,
repeatedly, as intentional ("NO NEW CHANNEL... CH_DRIVE carries the
Rademacher coefficients a"), and anticipated by R1's own kill-condition text
("diagnose by linear probe on log a (should be near-exact)"). An attack that
fired on raw per-step `a_i` alone would be attacking a documented,
intentional observational task. This file draws the line one order up: raw
order-0 gate values, fine; **composed** order-≥1 quantities, not.

**Class.** MISTAKES.md **D** (design-level failure). A **sharper instance of
D-2** ("an oracle that is the arm's own resolvent"). D-2's rule — "write down
what computes the label and what computes the prediction; if they are the
same operator, the comparison is void" — is extended here from a shared
OPERATOR to a shared CHANNEL: the comparison is equally void if the label's
computed intermediates leak into the compared object's input tensor, even
when the operators themselves differ.

**File.** `tests/mars_v15/test_skyline_gate_containment.py`

**Status, actually run — GREEN today, not a SKIP.**
```
test_raw_per_step_gate_is_present_by_documented_design_not_a_leak PASSED
test_composed_order1_signal_is_absent_from_x_on_the_live_bedm_corpus PASSED
test_containment_census_MUST_FIRE_on_a_planted_composed_leak PASSED
test_containment_census_MUST_FIRE_on_a_rescaled_planted_leak PASSED
test_composed_signal_absent_from_x_fed_to_the_v15_skyline SKIPPED
  reason: no v15 skyline function found (tried ceq.beds.bed_m.skyline /
  gated_scan_skyline, scale.arm_pl.skyline). BED-M's skyline has not been
  built yet (V15_LEDGER.md NEXT=it.3-4).
```
`scale/negation_scope.py`'s chain family exists on disk today (unlike BED-K
and the interventional channel, which the contract schedules as fresh
builds), so the composed-signal containment census runs for real against it
rather than skipping — and reads clean.

**Must-fire, verbatim** (`python tests/mars_v15/test_skyline_gate_containment.py`):
```
demo OK: live chain corpus, 64 (channel,position) live-band pairs checked, 0
contain the composed order>=1 signal; planted leak at channel 15 detected at
4 pair(s)
```
The composed oracle signal (the running resolvent at every prefix length,
obtained through the corpus's own public `equilibrium_hop_reading` rather
than by naming a channel — "a real containment/correlation check on actual
tensors, not an assertion about a variable name") was planted verbatim into
an unused channel and, separately, as a rescaled affine transform
(`2.5·oracle − 0.3`, to prove the R² probe fires and not only the
exact-match branch); both were detected and both raised
`AssertionError: composed oracle signal is recoverable from x`. The census
correctly stayed silent on the same corpus's raw, intentionally-observable
`a` channel (separate control test) and correctly restricted itself to the
live band (positions past `head`), where the chain's own zeroed prefix would
otherwise make a trivial `z=b` match register as a false positive.

---

## Attack 4 — Guards chosen after seeing itineraries

**Mechanism, one sentence.** BED-1 labels by committor and defines
"guards = `q=1/2` surfaces; itinerary = actions," admissible when the Pesin
deficit (`λ̂ − h_sym`) reads near zero — but if a guard surface was picked or
tuned *after* the itinerary (the symbol sequence) it is scored against
already existed, a small deficit is evidence the guard was fit to its own
answer key, not evidence the guards form a genuine generating partition.

**Class.** MISTAKES.md **V** (vacuous control). Closest existing entries are
**M-7** ("a pre-registration with a hole," whose own fix was disclosing a
row's timestamp against the data's) and **D-4** ("registration without
admission"). **New mechanism**: neither existing entry checks *ordering*
between two dependent artifacts (a guard and the itinerary it is scored
against); this is the first instrument in MISTAKES.md's history keyed on
journal-timestamp precedence as a hindsight-bias falsifier.

**File.** `tests/mars_v15/test_guard_itinerary_timestamp_order.py`

**Status, actually run.**
```
test_ledger_carries_zero_itinerary_events_today PASSED
test_bed1_guards_do_not_postdate_their_itinerary SKIPPED
  reason: BED-1 not built: 0 guard-definition event(s), 0 itinerary
  event(s) in house-events.jsonl. V15_LEDGER.md NEXT=it.3-4, BED-1 heads
  scheduled it.18-21.
test_paired_violations_MUST_FIRE_on_a_guard_that_postdates_its_itinerary PASSED
test_paired_violations_clean_case_does_not_fire PASSED
test_bed1_guard_and_itinerary_events_MUST_FIRE_scoping_finds_a_planted_pair PASSED
```
Measured directly against the live `house-events.jsonl` via `scale/ledger.py`
(parsed, never grepped, per the standing rule — `json.dumps`'s
space-after-colon silently defeats a literal grep on this exact file):
10,810 total events, 23 `guard`-word hits (all unrelated R10-era noise),
**0** `itinerary`-word hits, **0** events matching both, **0** BED-1 /
isocommittor / `q=1/2` hits.

**Must-fire, verbatim** (`python tests/mars_v15/test_guard_itinerary_timestamp_order.py`):
```
live ledger: 10810 events, 0 BED-1 guard event(s), 0 BED-1 itinerary
event(s) -- SKIP (BED-1 not built)
demo OK: planted guard-postdates-itinerary pair correctly raised
```
A synthetic guard event timestamped one hour after its paired itinerary
event (shared `id`) produced one violation and raised
`AssertionError: ...guard-definition artifact(s) postdate the itinerary they
are scored against...`; the mirror case (guard one hour *before* its
itinerary) produced zero violations and did not raise. A third fixture
confirmed the BED-1/isocommittor scoping itself finds a planted pair inside
synthetic events carrying unrelated `guard`-word noise, so the real test's
SKIP is shown to be about absence in the live ledger, not about the finder
being unable to see anything (MISTAKES.md V-7).

---

## Which attack is most likely to land when R1 runs

**Attack 2 (parity by underpowering).** Reasoning:

- It is not contingent on any implementation bug. Attacks 1 and 3 require
  something to have gone wrong in ARM PL's reading step or in a shared
  buffer between the skyline and `x`; Attack 2 fires from arithmetic alone —
  at N=8, achieved power against a real half-sigma effect is ≈23% by
  construction, regardless of how carefully everything else is built.
- It is not scoped to R4. PART IV requires every R-cell, R1 included, to
  report "distance-to-native-skyline" alongside its floor verdict, and PART
  I's resolution-statement clause is general to any comparison reported
  below N=23 — so R1's own skyline-distance column is a plausible site for a
  resolution-statement-shaped row days before R4 exists.
- It is independently corroborated by a document already in the tree
  (`V15_CONTRACT_ARITHMETIC_AUDIT.md` §A-1) computed from the contract's own
  numbers, not from this filing — and the coordinator's own `V15_LEDGER.md`
  NEXT block already names Mars's attack #2 as the destination for that
  finding.
- Attack 4 is structurally inapplicable at R1: BED-1 is scheduled it.18-21,
  nine iterations of critical path after R1. Attacks 1 and 3 are plausible
  but contingent; Attack 3 in particular already reads clean on the corpus
  most likely to be reused, which is evidence (not proof) it will stay
  clean.
