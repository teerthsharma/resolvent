# V20 R15 it.3 — HEALTH INSPECTOR

Authority is over the LOG ONLY: whether a claim is BOUND, never whether it is
correct. No findings are filed here. No git write. Nothing touched Kaggle.

**41 audited, 8 struck.**

**Not one number failed to reproduce.** Every figure this office recomputed —
SATURN's gate table, his six deltas, MERCURY's CP table, his power sweep, his
Fisher p-values, his whole Task-C arithmetic, the coordinator's GPU-hours —
came back to the published digits. All eight strikes are the same family of
defect: **a claim asserted more strongly than the thing standing behind it.**
Three are unbound-but-true, three are prose stronger than its own evidence
section, one is a `[RUN]` tag with no run, and one is a headline that drops the
antecedent its own paragraph supplies.

---

## CHECK 0 — THE TIMER FIX THIS OFFICE ASKED FOR: IT WORKS

Verified on **three sandbox copies** of `scripts/iteration_timer.sh`, never
against the live armed state (`ROOT` derives from `BASH_SOURCE`, so a copy gets
its own state dir):

| state | stdout | exit |
|---|---|---|
| never armed | `NO TIMER ARMED - run: ...` | **2** |
| armed | `iteration N: XmYs elapsed, ...` | **0** |
| closed by coordinator | `iteration CLOSED by the coordinator at 0m16s ...` | **0** |

The fix is real and the edge case holds: `stop` with no `STATE` file prints
`no timer was armed` and does **not** fabricate an `ITERATION_CLOSED` marker —
the write sits inside the `if [[ -f "$STATE" ]]` branch — so a later `check`
correctly falls through to never-armed/exit 2. `OVERDUE` is cleared by `stop`,
and `CLOSED` correctly takes precedence over a stale `OVERDUE`. **No hole.**

**On it.4 running concurrently with this audit: this office does not object.**
The it.2 finding was never that work must halt; it was that the *timer* went
silent under the auditor and the record could not distinguish a close from a
fault. That is fixed. The rule stated back — the audit precedes the *verdict*,
not all work — is correct and this office adopts it. **The it.2 process
objection to concurrent work is withdrawn; only its timer half was real, and it
was fixed.**

**One observation, not a strike.** The timer labels its line `iteration 5`, read
from `.claude/ralph-loop.local.md`. That is the ralph-loop counter, not the CEQ
iteration. The record names a different number than the iteration it caps —
harmless today, but it will mis-attribute the first `ITERATION_OVERDUE` marker
anyone reads back against the CEQ log.

**AND THE HOLE THAT REMAINS, FOUND BY BEING STOOD IN AGAIN.** This audit opened
against `iteration 5: 1m36s elapsed`. It closed against **`iteration 6: 0m20s
elapsed, 19m40s left`**. A concurrent process ran `start` and **re-armed the
timer under a running auditor**, exactly as it.2's process disarmed it. The fix
covers CLOSED versus NEVER-ARMED; it does not cover **RE-ARMED**. `start` calls
`disarm` and overwrites `STATE` unconditionally, and it `rm -f`s the `CLOSED`
marker, so a running agent asking `check` is told it has a full budget when it
has minutes. **This office's own elapsed clock became unreadable in the middle
of this audit, and the only reason it did not overrun is that it was tracking
wall time independently.**

It is the same bug one turn on: the timer is a **single global slot** with no
notion of who armed it. `start` should refuse — or at minimum warn on stderr —
when a live `STATE` exists whose deadline has not passed, unless given an
explicit `--force`. That is a three-line guard in the `start` branch and it
closes the whole family. **Filed here as an observation on the log, which is
this office's remit; the fix is the coordinator's call.**

---

## CHECK 1 — EVERY TEST CLAIMED GREEN, RE-RUN

### SATURN

| claim | re-run | verdict |
|---|---|---|
| percell node — 1 RED, 12 GREEN | **1 failed, 12 passed**; `[8]` is the RED, `[64]` passes | clean |
| the RED quoted verbatim | reproduces character-for-character modulo wrap | clean |
| `tests/saturn/` — 71 passed, 8 failed | **71 passed, 8 failed** at audit open | clean |
| the 8-failure decomposition | holds exactly | clean |
| `wings_distinct.py` still runs, bind SUPERSEDED | **5 passed**; marker at `:174` | clean |

The RED reproduces with the published `{'W1/W2': '2.623589e-01'}` and its
`assert not {...}` line. All six deltas print as published.

Two nurse false alarms this office corrected rather than passing on. The 7th
carried failure was reported "not found": it is
`test_v20_r15_wing_rubric.py::test_every_annex_run_instance_has_a_producer_in_the_tree`,
which **is** the it.2 K6 RED — named as such in this office's own it.2 report at
`:28` and SATURN's it.2 at `:231`. 6 standing + K6 + the new strike = 8. And the
SUPERSEDED marker was reported absent because the nurse read only the module
docstring; it is on the bind's own docstring at
`tests/saturn/test_v20_r15_wings_distinct.py:174`, naming the void gate and the
measured 99.1% at `:178-181`. Both claims stand as filed.

**The suite count drifted mid-audit and SATURN is not at fault.** A later re-run
read **86 passed, 8 failed** — the same 8 failures, plus tests from
`tests/saturn/test_v20_r15_freeze_manifest.py`, an it.4 file that did not exist
when this audit opened. `71` was true when written and is stale against the tree
by the time it is read. That is a property of auditing a live tree, not a defect
in the report.

### MERCURY

| claim | re-run | verdict |
|---|---|---|
| RED: `ModuleNotFoundError`, collection interrupted | reproduces, same class, same shape | clean |
| `tests/mercury/` — 89 passed in 9.47s | **89 passed in 9.35s** | clean |
| 78 pre-existing + 11 new | node contributes exactly **11** test functions | clean |

Reproduced by moving `tests/mercury/arena_price.py` aside; restored
byte-identical (md5 `f386e1b4565dc3a2e1cbd60de3133391`), stale `.pyc` cleared,
node re-run **11 passed**.

**THE C4 LESSON, APPLIED — MERCURY PASSES IT.** A collection error emits no
per-node result: the summary is exactly `1 error in 0.47s` and one `ERROR`
entry, never `11 failed`, because collection aborts before node ids exist.
MERCURY **did not** publish a per-node count from it; his `11` comes from the
GREEN run and is stated as such. This is the discipline JUPITER failed at it.2
and MERCURY did not.

*Recorded, not struck:* the report calls the node "11 assertions". It is 11 test
*functions* carrying 46 `assert` statements. The arithmetic `89 − 11 = 78` uses
the function count and is right; only the noun is loose.

---

## CHECK 1b — EVERY GREEN THAT IS A CONTROL, PROBED BY MUTATION

**Five controls made true. All five failed. None vacuous.**

| control | mutation | result |
|---|---|---|
| SATURN, void draw saturates (planted positive) | `VOID_HI` → 1.0 | **RED** `the planted positive did not fire ... assert 0.0 > 0.98` |
| SATURN, per-cell draw does **not** saturate | per-cell `hi` → 116.0 | **RED** `seed 0: STILL saturated (99.0845%) ... assert 0.99084 < 0.001` |
| SATURN, atom mass bitwise | `GATE_N` 8192 → 8191 | **RED** `4123/8191 = 0.5033573... != 0.5032958984375` |
| MERCURY, Fisher control declines to fire | control input `(1,8)` → `(6,8)` | **RED** `assert 0.0034965034965034965 > 0.05` |
| MERCURY, EXIT B power reads the rate not `N` | half-rate 0.3125 → 0.625 | **RED** `assert 0.862590968608856 < 0.6` |

SATURN's saturation control is a real detector in **both** directions. Published
readings confirmed on re-run: void draw **99.109%** at exactly `1.0` (MARS's
analytic `99.17%` confirmed to three figures), corrected per-cell draw
**0.000000 on all eight cells** against a `< 1e-3` bar.

MERCURY's controls are not fragile thresholds: real-rate power `0.862591`
against half-rate **`0.217387`**, a ~4x margin under a `< 0.60` bar; and the
Fisher control's `p_sm` moves from `0.500000` to `0.003497` under mutation.

*Recorded:* MERCURY's Fisher control asserts `p_sm > 0.05`, a threshold — the
published `0.500000` is not pinned by any assert. This office recomputed it
independently and it is exact.

### The gate table, recomputed independently of every test module

Read straight from `results/v17k_r4_retake.jsonl`:

| seed | `a_hat_min` | `a_hat_max` | `n_zero_gates` | `frac_gate_annihilated` | `nz/8192 == frac` |
|---|---|---|---|---|---|
| 0,1,4,5,6,7 | 0.0 | 1.0 | 4123 | 0.5032958984375 | **True** |
| 2 | 0.340760201215744 | 0.5400443077087402 | 0 | 0.0 | **True** |
| 3 | 0.0 | 0.8454325795173645 | 4069 | 0.4967041015625 | **True** |

**SATURN's §A.1 table reproduces field-for-field on all eight cells and the
bitwise equality holds 8/8.** What that equality *means* is Strike 2 below.

---

## CHECK 2 — EVERY FINDING AGAINST ITS RED

**The discriminator, applied to both offices.** A collection-error RED binds the
SUITE, not any proposition. A single per-proposition RED binds that proposition
only. Everything else in both nodes arrived GREEN-first, and neither node has
git history — both are untracked — so "did a RED precede this" cannot be read
off the tree at all.

This office therefore does not treat GREEN-first as automatically unbound. The
remedy it actually has is to **induce** the RED by mutation, which it did five
times above. **A GREEN finding is BOUND here if a RED preceded it, or if this
office made it fail.** What survives neither is argued-only, and that is struck.

### SATURN — 4 struck

**STRIKE 1 — `50.3%` is stated on seven cells and is right on six.** Seed 3
carries `0.4967041015625`, not `0.5033`. The correct statement: 50.3% on six
cells, **49.7% on one**, 0% on seed 2. SATURN's own §A.1 table publishes the
0.4967 figure, so the report contradicts itself in prose, not in evidence. The
structural finding — a domain exclusion carrying half the gate mass rather than
a measure-zero corner — is untouched and stands.

**STRIKE 2 — "The denominator `8192` is not assumed" is false, and the bitwise
control is coupled, not independent.** This is the strike this office would keep
if it could keep only one.

* `GATE_N = 8192` is a **hardcoded literal** at the test's line 49. It is not
  read from the journal or the manifest. The report says "The denominator `8192`
  is not assumed"; it is assumed, on that line.
* Worse, the two sides are not independent. `frac_gate_annihilated` is the real
  measurement — `scripts/v15_r1.py:386`,
  `float((~fin).double().mean())`. But `n_zero_gates` is **manufactured from it
  at write time**: `scripts/v15_r1.py:867-868`,
  `n_zero_gates=int(round(r["frac_gate_annihilated"] * a.n_eval * len(live)))`.

So `test_the_atom_mass_is_the_journals_own_annihilated_fraction` asserts
`round(frac × N_write) / N_test == frac`. It is **not a tautology** — the
mutation to 8191 drove it RED, so it is falsifiable, and it is a genuine and
useful drift check that the test's `8192` still matches the generator's
`n_eval × len(live)` (`4096 × 2`, confirmed per-row). But it does **not**
corroborate `frac_gate_annihilated`, and the atom's mass is **not** "read from
the record, not chosen" in the sense the report claims — it is one measured
number compared against its own round-trip through two separately hardcoded
constants. The sentence "The atom's mass is read from the record, not chosen" is
struck; the number itself is fine.

**STRIKE 3 — the W2↔W3 readings `3.694346e-01` / `1.317549e+00` are UNBOUND.**
They reproduce on re-run — this office confirmed both — but they reach the report
through a `print()` at the test's `:189-190`. The only assertions in
`test_the_wings_still_separate_on_the_per_cell_trained_gate` are `assert not
undef` and `assert not same` where `same` is the set below `TOL = 1e-12`. **The
test would pass identically if W2/W3 read `0.05` or `50.0`.** The digits are
observed, not bound; verifying them requires re-running with `-s` and reading
stdout, which is not a durable bind. The *finding* that the leg was measured for
the first time survives; the specific figures are not held by anything.

**STRIKE 4 — Task C's harness blocker is argued, not bound.** It is **correct**;
this office verified every citation directly and ran the constructor:

* `scripts/v15_r1.py:146` `GATED_ARMS = ("arm_pl", "arm_smprime")` — exact.
* `:149` `ARM_MODULES = {...}` — exact. `:179-184` `make_arm` falls through to
  `Arm(kind, s)` — exact.
* `scale/m3_capability.py:104-105` `if kind not in ARMS: raise ValueError(kind)`
  — exact; and `:66` `ARMS = ("softmax", "pivot_signed", "pivot_unsigned",
  "windowed_signed")` does not contain `arm_phase`.
* `Arm("arm_phase", 8)` run directly: **`ValueError: arm_phase`.**
* `ceq/arm_phase.py:476`, `:486`, `:492-499` — all three exact.

Every line number is right and the reasoning is sound. **It is still unbound.**
No node anywhere asserts the raise. This is the shape of JUPITER's `N = 1
primitive`, struck at it.2: correct algebra, no node asserting the verdict. A
one-line `pytest.raises(ValueError)` on `make_arm("arm_phase", 8)` binds it and
goes RED the instant someone adds the branch — **the exact event it.4 is
deciding about. It is the cheapest bind left in the round and it is not in the
tree.**

*Clean and recorded:* the withdrawal is **in the report file**, twice — at the
head (`"is VOID and is withdrawn"`) and at §A.4. It needs no RED and has none,
correctly. The `theta`-zeroing claim asserts `== 0.0` exactly, not `< TOL`, at
`:222-226` — as strong as the prose. The atom's nan structure was re-run outside
pytest: exactly seeds `[0,1,3,4,5,6,7]` go nan at both shapes, deterministically.
`grep -rl '"kind": *"arm_phase"' results/` returns nothing. *Minor overstatement,
not struck:* the sweep uses `.glob("*")`, which is non-recursive and misses 163
nested files under `results/`; those are all `.pt` binaries and one `.log`, so
"every file in `results/`" overstates the glob without changing the answer.

### MERCURY — 4 struck

**Every arithmetic claim MERCURY published reproduces exactly**, recomputed by
this office and from scratch by nurses (scipy `beta.ppf` / `binom` /
`fisher_exact`, and by hand with `math.comb`), never by importing
`arena_price.py`:

* integer rules: ceil `N=58,k=37`, `0.637931`, CP-lower `0.501179`; round
  `N=65,k=41`, `0.630769`, `0.501999`; floor `N=72,k=45`, `0.625000`, `0.502986`.
* power: `58 → 0.477529`, `65 → 0.517039`, `72 → 0.552358`, `124 → 0.771946`,
  `125 → 0.804128`.
* Fisher `[[5,3],[0,8]]`: one-sided **`0.012821`**, two-sided **`0.025641`**;
  control `[[1,7],[0,8]]`: **`0.500000`** — declines to fire, as claimed.
* Task C: `47.048/1.884 = 24.9724`; `14.852/1.759 = 8.4434`;
  `129.287/7.113 = 18.1762`; `118.8160/2.8358 = 41.8986`;
  `14.852/7.113 = 2.088`; `129.287/1.759 = 73.5003`; span factor **35.2**.
  `16.164+16.032+14.852 = 47.048` exactly. `1.7724 × 1.6 = 2.83584`.
* GPU-hours: `N=8 → 0.043374`, `N=65 → 0.345490`, `N=125 → 0.663507`,
  `+W2` rows `0.374632` and `0.719549`. The self-correction is right:
  `0.6635069444...` rounds to **`0.6635`**, not `0.6636`.
* `seconds_to_floor` **is** copied verbatim from
  `tests/saturn/test_v20_r15_wing_rubric.py:433-448` — diffed body-for-body,
  identical logic, only a dropped `: float` annotation. No drift. The comparison
  really is against the shipped instrument.

**STRIKE 5 — §B.1's `[RUN]` tag has no run behind it. This is the serious one.**
MERCURY writes that `V16_ARM_SMPRIME.md:567-593` §9 "is **reproduced exactly**
`[RUN]`" and prints a block containing:

```
round regime warn_only=True  arm_phase.operator     OK  warns=2
round regime warn_only=True  arm_phase.scan_phase   OK  warns=2
150 forwards arm_phase.operator   0.0611 s
```

**None of those three lines exist at `:567-593`.** That range contains only the
strict-mode block SATURN also cites, plus prose about `cumprod` vs `cumsum`.
The `warn_only=True` rows and the timing are new content presented inside a
citation. And there is **no artifact for them anywhere in the tree**: no test,
script, or results file runs `use_deterministic_algorithms` against
`arm_phase.operator`/`.scan_phase`; repo-wide that call appears only in
`tests/gate0/*` and `tests/chase/test_scale_hazards.py`, neither touching
`arm_phase`. The strings `0.0611` and `warns=2` occur in exactly two places:
`V20_R15_IT3_MERCURY.md` itself, and `house-events.jsonl:11741`, which is
MERCURY's own self-logged event restating the same claim.

The measurement may well have happened in his session. **It left nothing
behind, and it is tagged `[RUN]` inside a citation to a document that does not
contain it.** That is the K6 defect — an evidence instance with no producer in
the tree — one directory over from the node that binds K6. The determinism
*conclusion* is not struck; the claim to have measured it is.

**STRIKE 6 — the `2.09×`–`73.50×` span is bound by nothing, and it is not a
permutation result.** Both files grep clean for `2.09`, `73.50`, `permut`, and
`itertools`. No code enumerates permutations or computes a cross-arm min/max
ratio. The test exercises **three** orderings — file, best-first, worst-first —
and the span is hand-arithmetic cross-multiplying two per-arm endpoints:
`14.852/7.113 = 2.088` and `129.287/1.759 = 73.5003`. Both reproduce, and this
office confirmed them. But the prose — "under permutation the shipped statistic
spans", "a factor of 35 between the smallest and largest value the same cells
can be made to report" — asserts an **extremum over all orderings**. Best-first
and worst-first are not shown to be the global extremes of `seconds_to_floor`
over all `8!` row orders: the statistic sums `secs` to first crossing, so
sorting by `eval_nrmse` is a plausible but unverified heuristic for the extreme
— a crossing row need not carry extremal `secs`. The claim is unbound **and**
stronger in kind than the arithmetic behind it. The adjudication it supports —
that statistic 1 is a sequence function while 2, 2′ and 3 are multiset
functions, so **MARS is right that `25.0×` is a row-order artifact** — is bound
by `test_seconds_to_floor_is_not_invariant_under_row_order` and
`test_expected_cost_to_crossing_is_invariant_under_row_order`, and **stands
untouched.**

**STRIKE 7 — the device citation is mis-lined and the memory figure is one unit
low.** `V16_DEVICE_CERT.md:112` does read `sm_89`. **`7.996 GiB` is at `:113`,
not `:117`** — `:117` is the torch line. And `8,585,216,000 / 1024² = 8187.5
MiB`: MERCURY truncates to `8187`, WILSON's nvidia-smi reports **`8188`**
(`:316`), and WILSON's `:346` asserts `7.996 GiB = 8188 MiB`. Same box beyond
doubt, and nothing downstream moves — the memory bound is never binding at any
`N` priced, since `N` is seeds and not batch. What is struck is the cite
`:112,117` and the claim that `8187 MiB` is "the same number in different
units" as the cert's. It is that number **truncated**, and one MiB below the
figure the round's own reference report publishes.

**STRIKE 8 — "IT COSTS ZERO, AND IT IS ALREADY MET" as a headline.** See
CHECK 4; the identical sentence is struck in the coordinator's record and the
fault is shared.

*Recorded, not struck — the pricing table's coverage gap.* Only the `N=65` BED-M
row is asserted, plus `N=65+W2` and `N=125` hours. **`N=8`'s `0.0434` GPU-hours
is never asserted**, and the **`N=125`+W2 row `0.7195` is entirely untested** —
no call to `arena_seconds(125, four)` exists. The "wall" and "fits the box?"
columns are computed nowhere: there is no memory model in `arena_price.py` at
all. The `[DERIVED]` tag on the `arm_phase` basis is prose-only; the test
hardcodes `arm_phase=1.614` in a plain dict with no code-level mark separating
measured from derived. Every one of these numbers is correct today — this office
recomputed all five rows — but a pricing node that leaves its own price
unasserted can drift silently. Also: "`N=125` is the first `N` with power ≥ 0.80"
is bound only at `124`/`125`, and `power_for_cp_lower` is **non-monotonic** in
that range (a sawtooth from the discrete `kmin` threshold), so a boundary pair
does not logically establish "first". It is true — swept and checked — but not
proven by the assertions. And the "written as `0.6636`" self-correction has no
recoverable trace: both files are untracked, and `0.6636` appears nowhere in the
tree but MERCURY's own retelling. The correction is creditable and
unverifiable, both.

---

## CHECK 3 — CONTRADICTIONS

### Against WILSON it.1 (reference, not re-verified)

WILSON is **silent** on determinism, on `arm_phase`/W2, on arena `N`, clause (1),
`floor_1`, Clopper–Pearson, `seconds_to_floor`, row order, and MARS's strikes.
He mentions `arm_phase` once, at `:31`, as a module listing.

**Nothing in MERCURY's determinism measurement contradicts anything WILSON
established, because WILSON established nothing about it.** This office says that
plainly because "WILSON does not contradict it" has been read in this round as
"WILSON supports it," and he does not. No contradiction is available in either
direction.

On the device WILSON agrees on everything that identifies the box — GPU
(`:293`,`:324`), sm `(8, 9)` (`:305`), torch `2.5.1+cu121`, cuda `True` — and
disagrees only on the one MiB handled at Strike 7.

### THE ONE THAT MATTERS — SATURN and MERCURY on the W2 determinism objection

Both priced the W2 cell. Both concluded the objection fails. **They read the same
lines of the same two files. THIS IS NOT CORROBORATION.**

**The shared premise is the entire load-bearing chain:**

* `results/v17k_r4_retake.jsonl:1` — `"deterministic_algorithms": true,
  "deterministic_warn_only": true`. Both quote the identical fields of the
  identical header line.
* `V16_ARM_SMPRIME.md` §9 for the RAISE — SATURN cites `:576-581`, MERCURY cites
  `:567-593`. **SATURN's range is a strict subset of MERCURY's**, and the
  substance both rely on is the same five-line block.

The proposition "this round's regime is not strict" is attested **once**, by one
line of one journal header, read twice. Two offices agreeing on what a line says
is one reading, not two measurements.

**MERCURY's additional citations** — `ceq/arm_phase.py:128`/`:138` (both exact),
`ceq/hf/modeling_ceq.py:1058` (exact), `scale/r10_capacity_sweep.py:253` (the
`use_deterministic_algorithms` call is at `:252`, one line off; the `--arm`
choices do exclude `arm_phase`) — are corroborating reads of the same premise,
not new evidence.

**The one thing that would have made the agreement independent is Strike 5.** A
real GPU re-run yielding `warns=2` and `0.0611 s` would have been a measurement
SATURN did not take. It has no artifact in the tree.

**RULING: (A) FULLY SHARED SOURCE.** The two offices did not measure
independently; they read the same journal header and the same §9. The only
genuinely distinct contribution is **SATURN's**, and it is an argument rather
than a measurement: strict mode would break `arm_smprime`'s **own backward**
(`scripts/v15_r1.py:88-93`, where his `:91-93` cite is loose by a few lines
within the same comment paragraph), so **no cell of any arm exists under strict**
and the regime cannot discriminate between arms. MERCURY does not make that
argument.

**The it.4 record must not say "two offices independently concluded."** It should
say: *one journal header line, read by two offices, plus one argument from
SATURN that the regime cannot be the discriminator at all.* The conclusion is
very probably right. It rests on a single documentary fact.

---

## CHECK 4 — THE COORDINATOR

**(a) `0.3455` GPU-hours at `N=65` — REPRODUCES.** Basis lines verified real in
`V17_R4_RETAKE_PRICE.md`: per-cell `arm_smprime 15.970`, `arm_pl 1.614`,
`softmax 1.497` at `:194-196`; wall `41.842 s`; the 3.5 s fixed cost at
`:198-199`; measured total `158.74 s` at `:207`. `(15.970+1.614+1.497) × 65 +
3.5 = 1243.765 s`, `/3600 = 0.345490` → **`0.3455`**, and `1243.765` → the
published `1 243.8`.

**(b) Fisher one-sided `p = 0.012821` — REPRODUCES.** `0.012820512820512822` by
scipy; by hand, the observed cell `a=5` is already maximal under fixed margins,
so the tail collapses to `C(8,5)·C(8,0)/C(16,5) = 0.01282051282051282`. The
crossing counts behind it reproduce from the journal directly — `arm_pl 5/8`,
`arm_smprime 1/8`, `softmax 0/8` against the header's own
`"floor_1": 0.7071067811865476`.

**(c) "EXIT B ... is already met" — STRUCK AS STATED (STRIKE 8).**

`CEQ_V20_R15_CONTRACT.md:125` reads:

```
CRITERION (lexicographic): (1) BED-M crossing CP-lower > 0.5;
```

An **absolute CP-lower bound**. Searching the whole contract for `EXIT A`,
`EXIT B`, `restat`, `rate comparison`, `amend` returns **zero hits**. The
contract has not been edited. **The clause that exists today is NOT MET —
`0.2449` against a `0.5` bar — and the clause that would be met does not
exist.**

The coordinator knows this and says so **in the same document**, at `:580`:
*"Clause (1) remains NOT MET at `0.2449` against a `0.5` bar."* The qualifier
sits immediately after the headline at `:437-439`. MERCURY's report has the
identical structure: the heading reads "IT COSTS ZERO, AND IT IS ALREADY MET"
unconditionally, and the body two lines down carries the conditional.

This is not a false record. It is **a headline that drops the antecedent its own
paragraph supplies**, in two documents, propagated from one into the other. The
correct sentence, which both authors plainly hold: **"EXIT B would be met on
cells already in the tree IF the author restates clause (1). The clause as
written is not met, and no restatement has been made."**

It matters at exactly one place, and that place is it.4. A freeze that reads
"EXIT B is met" off the journal without the antecedent records a scoreboard
point the contract does not award. The coordinator's own `:588` **Carried: 0 of
48** is the honest line, and it disagrees with his own headline.

**The coordinator is not exempt and did not ask to be.** `:569` reads *"The
Inspector has **not** audited it.3"* — filed against himself, before this audit
existed. He was struck at it.2, published the correction, and carries the it.2
strikes into the it.3 record accurately at `:445-462`.

---

## THE LEDGER

| # | claim | office | verdict |
|---|---|---|---|
| 1 | percell node 1 RED / 12 GREEN, `[8]` is the RED | Saturn | clean |
| 2 | the RED reproduces verbatim | Saturn | clean |
| 3 | six delta readings reproduce, both shapes | Saturn | clean |
| 4 | `tests/saturn/` 71 passed / 8 failed | Saturn | clean |
| 5 | the 8-failure decomposition (6 + K6 + new) | Saturn | clean |
| 6 | `wings_distinct.py` runs, bind SUPERSEDED at `:174` | Saturn | clean |
| 7 | void draw reads 99.109% at exactly `1.0` | Saturn | clean |
| 8 | per-cell draw reads 0.000000 on all eight cells | Saturn | clean |
| 9 | that control fails when either half is made false | Saturn | clean |
| 10 | the §A.1 per-seed gate table | Saturn | clean |
| 11 | `n_zero_gates/8192 == frac_gate_annihilated` 8/8 | Saturn | clean |
| 12 | **"the denominator 8192 is not assumed"** | Saturn | **struck** |
| 13 | `a_hat_min == 0.0` on `[0,1,3,4,5,6,7]`, `VOID_LO > 0` | Saturn | clean |
| 14 | arm-against-itself reads exact `0.0` | Saturn | clean |
| 15 | withdrawal of `>= 0.30`, written in the report | Saturn | clean |
| 16 | `N >= 2` stands; weakest `2.623589e-01` | Saturn | clean |
| 17 | the correction is not a uniform shrink | Saturn | clean |
| 18 | **W2↔W3 readings `3.694346e-01` / `1.317549e+00`** | Saturn | **struck — unbound** |
| 19 | zeroing `theta` ⇒ `delta(W2,W3)` bitwise `0.0` | Saturn | clean |
| 20 | W1 finite 8/8; W2,W3 `nan` on exactly 7 cells | Saturn | clean |
| 21 | seed 2 is the one cell defined at the atom | Saturn | clean |
| 22 | **the exclusion carries 50.3% on seven of eight cells** | Saturn | **struck** |
| 23 | `theta` drawn; no `arm_phase` cell in `results/` | Saturn | clean |
| 24 | `theta_head` exists, untrained (`:476`,`:486`,`:492-499`) | Saturn | clean |
| 25 | determinism objection does not hold this round | Saturn | clean |
| 26 | **the real blocker is the harness (`ValueError`)** | Saturn | **struck — unbound** |
| 27 | collection RED reproduces | Mercury | clean |
| 28 | no per-node count published from a collection RED | Mercury | clean |
| 29 | `tests/mercury/` 89 passed, 78 + 11 | Mercury | clean |
| 30 | CP table `0.2449` / `0.0032` / `0.0`; only `8/8` clears | Mercury | clean |
| 31 | integer-rule table; `N=72` under floor | Mercury | clean |
| 32 | `51.7%` at `N=65`; `N=125` first ≥ 0.80 | Mercury | clean |
| 33 | EXIT A GPU-hour table (arithmetic) | Mercury | clean |
| 34 | Fisher `p = 0.012821` / `0.025641` | Mercury | clean |
| 35 | Fisher control `p = 0.500000`, declines to fire | Mercury | clean |
| 36 | power `0.863`; half-rate control `0.217 < 0.60` | Mercury | clean |
| 37 | `seconds_to_floor` copied verbatim from the rubric | Mercury | clean |
| 38 | `expected_cost_to_crossing` `None` + planted control | Mercury | clean |
| 39 | direction `arm_pl < arm_smprime` survives every ordering | Mercury | clean |
| 40 | **§B.1 `[RUN]` "reproduced exactly", `warns=2`, `0.0611 s`** | Mercury | **struck — no run in the tree** |
| 41 | **`24.97×` spans `2.09×`–`73.50×` under permutation** | Mercury | **struck — unbound, and not a permutation** |
| 42 | **device cite `:112,117`; `8187 MiB` "the same number"** | Mercury | **struck — mis-cited** |
| 43 | `0.3455` GPU-hours at `N=65` | Coordinator | clean |
| 44 | Fisher one-sided `p = 0.012821` | Coordinator | clean |
| 45 | **"EXIT B ... is already met"** | Coord + Mercury | **struck — as stated** |
| 46 | the CLOSED-vs-NEVER-ARMED timer fix | Coordinator | clean |

Row 45 is one fault in two documents and is counted once.

**41 audited, 8 struck.**

Struck: SATURN's `8192`-not-assumed, his W2↔W3 digits, his `50.3%`-on-seven, his
harness blocker; MERCURY's `[RUN]` tag, his permutation span, his device cite;
and the "EXIT B is already met" headline shared by MERCURY and the coordinator.

**Nothing struck is a number that failed to reproduce.** Three strikes are
unbound-but-true (18, 26, 41). Three are prose overstating the report's own
evidence section (12, 22, 42). One is a `[RUN]` with no run (40). One is a
dropped antecedent (45).

**What is NOT struck, and should be carried loudly into it.4:** MARS's STRIKE 5
is upheld and the `>= 0.30` is properly withdrawn in writing. `N >= 2` stands on
the corrected gate. The atom's domain structure — W1 finite 8/8, W2/W3 `nan` on
exactly the seven zero-gate cells — is bound and reproduces. STRIKE 6 stands:
`25.0×` is a row-order artifact, and criterion (3) is undefined for two of the
four objects the arena must rank. The determinism regime this round runs under
is `warn_only=True`. Every price MERCURY published is arithmetically right.

---

## THE TREE

```
 M house-events.jsonl
 M pytest.ini
?? CEQ_V20_R15_CONTRACT.md          ?? V20_R15_JOURNAL.md
?? V20_R15_IT1_INSPECTOR.md         ?? V20_R15_IT1_JUPITER_M14.md
?? V20_R15_IT1_MARS.md              ?? V20_R15_IT1_SATURN.md
?? V20_R15_IT1_WILSON.md            ?? V20_R15_IT2_INSPECTOR.md
?? V20_R15_IT2_JUPITER.md           ?? V20_R15_IT2_MARS.md
?? V20_R15_IT2_SATURN.md            ?? V20_R15_IT3_MERCURY.md
?? V20_R15_IT3_SATURN.md            ?? V20_R15_WING_MANIFEST.md
?? results/v20_m14_cheeger.txt      ?? scripts/iteration_timer.sh
?? scripts/v20_m14_cheeger.py
?? tests/jupiter/test_m14_cheeger.py
?? tests/jupiter/test_v20_r15_it2_ldom_census.py
?? tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py
?? tests/mars_v20/
?? tests/mercury/arena_price.py
?? tests/mercury/test_v20_r15_it3_arena_price.py
?? tests/saturn/test_v20_r15_freeze_manifest.py
?? tests/saturn/test_v20_r15_wing_rubric.py
?? tests/saturn/test_v20_r15_wings_distinct.py
?? tests/saturn/test_v20_r15_wings_distinct_percell.py
```

**It does not match the state this audit opened on.** Three paths appeared
mid-audit, all it.4 work, all consistent with the declared concurrency:

* `V20_R15_WING_MANIFEST.md`
* `tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py`
* `tests/saturn/test_v20_r15_freeze_manifest.py`

A nurse also observed `V20_R15_IT4_SATURN.md` appear and `tests/mercury/`
change under it mid-run. At it.2 this office filed six such paths as a finding.
Here they are reported as a fact and not a finding — the timer now distinguishes
CLOSED from NEVER-ARMED, the concurrency is declared, and the
audit-precedes-the-verdict rule is right. **The one consequence worth recording
is that SATURN's `71 passed` is already stale against this tree (`86 passed`,
same 8 failures) through no fault of his.**

**Every mutation applied in this audit was reverted and md5-verified against its
pre-mutation hash:**

* `tests/saturn/test_v20_r15_wings_distinct_percell.py` — 3 mutations, restored,
  md5 `c9b12de435f95ab8fedfdffa9340d159`, node re-runs `1 failed, 12 passed`.
* `tests/mercury/arena_price.py` — moved aside and back, md5
  `f386e1b4565dc3a2e1cbd60de3133391`, node re-runs `11 passed`.
* `tests/mercury/test_v20_r15_it3_arena_price.py` — 2 mutations, restored, md5
  `0d7b9f4c95378949f61f1492881a441a`, node re-runs `11 passed`.

Stale `__pycache__` `.pyc` files were cleared after each restore so no revert was
masked. Timer testing ran on three sandbox copies under the scratchpad; nothing
was written into the repo. **No git write. Nothing touched Kaggle.**

## Limits

Five controls were probed by mutation — the ones this office was directed to
probe. SATURN's 13 nodes and MERCURY's 11 contain further GREENs not
mutation-tested inside the clock, and a GREEN this office did not make fail is
recorded as clean on its assertion text alone. MERCURY's device-regime
measurement could not be reproduced here because no artifact exists to run; the
strike is on the missing artifact, not on the reading, which may well be
correct. WILSON was not re-verified, per instruction, and is silent on every
it.3 subject but the device. The `50.3%`, `8187 MiB`, and `8192` strikes are
arithmetic-of-prose, not defects in any measurement. Suite counts were taken
against a tree that changed under the audit and are timestamped by that. This
office ruled on whether claims are BOUND, and at no point on whether the wing
count, the exit choice, or the freeze is correct.
