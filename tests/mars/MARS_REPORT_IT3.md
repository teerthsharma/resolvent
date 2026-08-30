# MARS / MORIARTY — R9 iteration 3, the restitution round

Fast-forwarded `5053616 → 486ae41` before touching anything.

**A thirteenth class survives.** Two instances re-verified by execution in this
worktree, neither taken from a report. Ruling on the coordinator's three-instance
question: **one class, and the drafted wording catches only one of the three.**

**Two GREEN attacks filed. One fires, one is refuted by its own planted null.**
Running total across three iterations: **12 filed, 9 fired.**

Tests: `tests/mars/test_mars_control_entry_point.py` (4) and
`tests/mars/test_mars_green_attacks.py` (6) — **10/10 pass, 29.21 s.**
No wall clock anywhere; every number is exact combinatorics or a pure re-run.

---

## 0. One existence claim corrected before it is used

The dispatch names *"twelve defect classes (X-R1..R12)"*. `git grep -n "X-R"`
over every `.md` and `.py` in the tree returns **one hit**, `CHECKLIST.md:556`,
which is unrelated (`MAX-ROW MECHANISM`). No `X-R1..R12` taxonomy exists in this
repository. The taxonomy that does exist is `MISTAKES.md`: **four families, 35
entries** — `V-1..V-13`, `P-1..P-8`, `M-1..M-9`, `D-1..D-5`. Everything below is
tested for reducibility against those 35, which is the half of the bar that has
a referent.

---

## 1. The class

> ### The control constructs its own input, so it certifies the instrument's predicate over a domain production never chose.
>
> **Rule.** A control must enter the instrument at the entry point production
> uses. If the control supplies its own subject, it tests the predicate and
> leaves the *selection* stage untested — and a selection stage that returns
> nothing then reads as "nothing is wrong" rather than "nothing was looked at".
> Drive the control through the production selector, or require the selector's
> output to be non-empty and to contain a planted item placed **in the world**
> rather than in the test.

### Instance A — `scale/chase_struck_coverage.py` at `8b40e16^`

Re-verified by importing that exact blob into this worktree and running it, not
by reading Jupiter's account:

```
pre-fix control() returned: True
ROOT = .../worktrees/agent-afb2a8ed4e030ff6a
candidates 375 | ABSOLUTE-parts filter (shipped) keeps 0 | RELATIVE-parts filter keeps 374
=== SCANNING 0 PATHS THE SHIPPED CHECK DOES NOT COVER ===
  scanned. uncovered .md: 0, uncovered .py: 0
=== NO STRUCK CONSTANT IS ASSERTED IN ANY UNCOVERED PATH ===
main() exit code = 0
```

The control passed, the tool looked at **0 of 375** files, and the exit code was
0. `control()` calls `scan_text` directly; `collect_targets` — the stage that
failed — never executes inside it.

### Instance B — `tests/cameron/test_published_intervals_have_producers.py`

Neptune reported this; re-verified here by execution rather than taken from the
report. The binder's subject list is the literal dict `PUBLISHED` at `:67`, and
an AST walk of the file finds **no** `read_text`, `rglob`, `iterdir` or `open`
call — there is no document-selection stage at all.

```
binder PUBLISHED keys: ['argmax - softmax', 'settled - softmax', 'settled - twin']
README.md                    distinct [lo, hi] pairs printed: 4
ceq/hf_artifact/README.md    distinct [lo, hi] pairs printed: 12
CHECKLIST.md                 distinct [lo, hi] pairs printed: 29
TOTAL 45   |   intervals the binder is parameterised over: 3
```

45 is an **upper bound** on the population — not every bracketed pair is a
bootstrap interval — but 3 is exact, and the mechanism does not depend on the
denominator. An interval added to any of the three files it names is neither
bound nor flagged.

### Instance C — the e-process must-fire battery (Deimos)

**Not counted toward the bar.** `READ` from the coordinator's summary and
commit `8616a41`'s body; not re-verified by me. Cited only as a third data
point on the shape question.

### Reducibility, entry by entry

| nearest | why this is not that |
|---|---|
| **V-6** *the branch under test never ran* | In all three the **asserted** branch ran: `scan_text` ran, the binder's `contrast` ran. V-6's rule — *"assert the code path under test executed"* — is satisfied. What did not run is a different, upstream stage the control never claimed to cover. V-6 asks "did your branch run"; this asks "did your input come from where production's comes from". |
| **V-7** *a reported absence needs a planted positive* | V-7 is **satisfied** in every instance. Each control was real, planted and non-degenerate. That is the whole point: V-7's rule was obeyed and the defect shipped anyway, because V-7 does not say **where** the positive must be planted. |
| **V-9** *a repair that changes nothing* | The repairs changed plenty. The defect is in the control, not the repair. |
| **V-13** *a search whose walk includes nested checkouts* | The **closest**, and it is the mirror image: V-13 is over-reach (the walk includes too much), instance A is under-reach caused by the same `.claude/worktrees/` fact. V-13 is about the walk's extent; this class is about whether any control observes the walk at all. |
| **P-1** *a number with no live producer* | There is a live producer in every case. |
| **M-7** *a pre-registration with a hole* | About what was promised, not about what the control touched. |

### Ruling on the coordinator's question: one class, two of three uncaught by the drafted wording

A, B and C are **one class**. The invariant that holds across all three is
"control input is constructed rather than selected", and each fails it.

Saturn's drafted type — *"the control that validates the matcher and never the
reach"* — is the right type stated in **search vocabulary**, and that vocabulary
reaches only instance A. Instance B has no matcher and performs no search; its
failed stage is "which documents exist". Instance C has neither. Applying the
drafted wording to B or C requires re-reading "matcher" as "predicate" and
"reach" as "domain", which is the generalisation, not the draft.

**That instance B arrived after the type was drafted and is not caught by its
words is the evidence requested.** The recommendation is that the type ship in
the generalised wording above, with A as the search instance and B as the
document instance, so the next occurrence in a third vocabulary is still caught.

### The mechanized catch, and the false alarm it produced

`tests/mars/test_mars_control_entry_point.py::entry_point_report` is a static
AST survey: a module owns a **selector** (it calls `rglob`/`glob`/`walk`/…, or is
named `collect_*` / `*_targets`) and a **control** (name contains `control` /
`must_fire` / `planted` / `red`), and the survey flags controls that do not reach
a selector through any intra-module call chain. Repo-wide over `scale/`:

```
=== CONTROL-ENTRY-POINT SURVEY: 1 module(s) whose control never reaches its own selector ===
  scale/chase_struck_coverage.py
      selectors ['collect_targets']   blind controls ['control']
```

**The first version of the gate asserted `blind == []` on that module and
failed — on a file that is correctly repaired.** `control()` at HEAD is
byte-identical to the pre-fix one; Jupiter's coverage control lives in
`tests/jupiter/test_struck_coverage_scans.py`, which calls `collect_targets` at
five sites including one that plants a file on disk. The invariant is not *"the
module's own control reaches the selector"* but *"the selector is driven by some
control"*. A catch that legislates the control's **location** instead of the
selector's **coverage** is the same mistake one step along, and it is recorded in
the shipped docstring rather than quietly rewritten.

The gate now asserts: every selector named by the survey is called from at least
one file under `tests/`. It ships with a RED (a synthetic pre-fix module the
survey must name, and its repaired twin the survey must clear) because a catch
that only ever saw the repaired file could not tell the two apart.

---

## 2. Attacks on the standing GREENs

### G1 — FIRES. The published interval endpoints are a function of the bootstrap RNG seed, not of the data

`scale/m3_synthetic_settled.py:193-200` resamples the **five** per-seed deltas
10,000 times and reads `reps[250]` / `reps[9750]`. The statistic has exactly
`C(9,4) = 126` atoms over `5**5 = 3125` ordered resamples, with masses computable
in closed form. For `settled − softmax` the 2.5% target lands in a gap:

```
lattice[6] = 0.065659   P(mean <= v) = 0.017920
lattice[7] = 0.066232   P(mean <= v) = 0.024320   <- README ships this
lattice[8] = 0.068181   P(mean <= v) = 0.025920   <- CHECKLIST ships this
lattice[9] = 0.071449   P(mean <= v) = 0.027520
```

Which atom `reps[250]` lands on is a `Binomial(10000, 0.024320)` coin flip.
Measured, `contrast()` re-run at bootstrap seeds 0..99, `n_boot=10000`:

| contrast | `ci_lo` over 100 seeds | `ci_hi` over 100 seeds |
|---|---|---|
| `twin − softmax` | `{0.100873: 96, 0.099945: 4}` | `{0.121920: 96, 0.121793: 4}` |
| `settled − softmax` | `{0.066232: 32, 0.068181: 40, 0.071449: 23, 0.071582: 5}` | `{0.147110: 70, 0.146537: 30}` |
| `settled − twin` | `{−0.048587: 10, −0.042903: 62, −0.042563: 27, −0.042208: 1}` | `{0.031557: 69, 0.031912: 31}` |
| `argmax − softmax` | `{−0.134115: 93, −0.135101: 7}` | `{−0.102204: 36, −0.102786: 36, −0.103194: 19, −0.103286: 9}` |

**The `argmax − softmax` row is the sharp one.** `−0.102204` is family A
(`ceq/hf_artifact/README.md`) and `−0.102786` is family B (`README.md`,
`CHECKLIST.md`). **The shipped Monte Carlo alone produces both, 36 seeds each.**
For that contrast the two families are not separated by the number, and
`test_the_two_families_are_not_the_same_estimator` is true at `seed=0` by choice
of seed.

**Scope, stated plainly.** Family B *is* the exact percentile — my enumeration
reproduces it independently of Saturn's — so no published number is wrong, and
**no verdict moves**: every endpoint keeps its sign at every seed tested. What
fires is that family A's six printed digits are not reproducible, on a statistic
whose exact value costs 3,125 sums against the 10,000 draws that replace it.

**F-green itself survives this attack.** `twin − softmax` is the one contrast
whose endpoints are lattice-stable: `0.100873` at 96/100 seeds, and the exact
enumeration returns `0.100873`. The `+0.111396 [+0.100873, +0.121920]` headline
is reproducible. The attack strikes its **siblings**, not it.

Relation to the coordinator's new datum: Neptune's `argmaxste − argmax` shift is
a **constant `1.06e-4` on both endpoints**. This mechanism moves endpoints by
whole lattice steps and moves them **independently** — see `settled − softmax`,
where `ci_lo` takes four values while `ci_hi` takes two. So Neptune's shift is a
genuinely different cause with the same symptom. **Two distinct mechanisms are
producing "two published families", and only one of them is an estimator
difference.**

### G2 — REFUTED by its own planted null, and kept

The attack was V-10 against the settled birth gate: `gate1_event`'s
discriminating contrast is `d_H(settled fixed point, ONE step)` against a
threshold, and a gate whose quantity cannot approach the threshold from below has
no rejection region. The planted null is `beta → 0`, which makes the settle the
identity. Measured at s=64, d=24, k=8, seeds 0-2, threshold `1e-6`:

| beta | `d_onestep` range | fired |
|---|---|---|
| 0.5 | 7.139e+00 .. 1.539e+01 | 3/3 |
| 0.1 | 2.348e-01 .. 4.556e-01 | 3/3 |
| 1e-2 | 2.135e-03 .. 4.417e-03 | 3/3 |
| 1e-3 | 2.115e-05 .. 4.383e-05 | 3/3 |
| **1e-6** | 2.113e-11 .. 4.379e-11 | **0/3** |
| **1e-9** | 0.000e+00 | **0/3** |

`d_onestep` scales as roughly `beta**2` and the gate stops firing. **The gate is
not vacuous.** One qualification the gate does not itself state: its rejection
region begins near `beta = 7e-4`, so it separates *"settling happens at all"*
from *"settling does not happen"* across four orders of magnitude of beta, and is
not evidence that settling is **material** at the shipped `beta = 0.5`. The null
it lacked is now in the tree.

### Not attacked

**M4 eviction, the E4′ gates and the calibrated M3 harness were not reached.**
No attack is filed against them and none should be recorded as surviving one.

---

## 3. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | no `X-R1..R12` taxonomy exists in tree | `RUN` | `git grep "X-R"` → 1 unrelated hit |
| 2 | `MISTAKES.md` holds 35 entries in V/P/M/D | `READ` | headers counted, 13+8+9+5 |
| 3 | pre-fix scanner: `control()` True, 0 of 375 kept, exit 0 | `RUN` | `8b40e16^` blob imported and executed here |
| 4 | relative filter keeps 374 of the same 375 | `RUN` | same run |
| 5 | binder is parameterised over exactly 3 intervals | `RUN` | AST of `PUBLISHED` |
| 6 | binder opens no document | `RUN` | AST call set ∩ {read_text, rglob, iterdir, open} = ∅ |
| 7 | the three documents print 45 distinct `[lo, hi]` pairs | `RUN` | regex; **upper bound** on the population |
| 8 | instance C (e-process battery) | `READ` | coordinator + `8616a41` body; **not counted** |
| 9 | the class is not V-6 / V-7 / V-9 / V-13 / P-1 / M-7 | `DERIVED` | entry-by-entry table, steps shown |
| 10 | survey flags exactly 1 module in `scale/` | `RUN` | `entry_point_report()` |
| 11 | my first gate was a false alarm on a repaired file | `RUN` | it failed; `collect_targets` has 5 call sites in `tests/jupiter/` |
| 12 | 126 atoms over 3125 resamples, all four contrasts | `RUN` | exact enumeration |
| 13 | exact masses bracket 0.025 at 0.024320 / 0.025920 | `RUN` | closed-form cumulative |
| 14 | `settled − softmax` ci_lo takes 4 values over 100 seeds | `RUN` | `contrast(seed=0..99)` |
| 15 | shipped MC produces **both** published `argmax − softmax` upper endpoints, 36/100 each | `RUN` | same sweep |
| 16 | `twin − softmax` is lattice-stable; F-green survives G1 | `RUN` | 96/100, exact == published |
| 17 | no verdict moves under G1 | `RUN` | every endpoint keeps its sign at every seed |
| 18 | `d_onestep` collapses as `beta → 0`; gate fires 0/3 at 1e-6 | `RUN` | `gate1_event` beta sweep |
| 19 | Neptune's `1.06e-4` shift is a different mechanism from G1 | `DERIVED` | G1 moves endpoints by lattice steps, independently; his is constant on both |

---

## 4. What I could not validate

Instance C is `READ` and is excluded from the bar, so the class stands on two
instances rather than three; if either A or B is later shown reducible, the class
falls to one and should be withdrawn. The reducibility table is `DERIVED` — it is
an argument about wording, and a reader who holds that "the branch under test" in
V-6 already means "every stage the instrument runs" can collapse this class into
V-6, and I cannot refute that reading by measurement. The 45-interval denominator
is an upper bound I did not adjudicate pair by pair; only the numerator, 3, is
exact. The survey scans `scale/` alone and matches control functions by name, so
a control called something else is invisible to it and the single hit is a floor,
not a census — it is shipped as a lead generator for that reason. G1's seed sweep
covers 100 seeds at the shipped `n_boot`; I did not compute the exact Binomial
tail probability that would predict the observed 32/40/23/5 split, so the
mechanism is `DERIVED` from the mass table and the frequencies are `RUN`. I took
no reading and touched no arm; `scale/m3_quintuple.py`, `scale/eprocess.py`,
`scale/capability_table.py`, `results/*.jsonl` and `MISTAKES.md` are untouched, so
the class is reported for Saturn to file rather than filed. **M4 eviction, the
E4′ gates and the calibrated M3 harness were not attacked at all** — the dispatch
asked for one attack per standing GREEN and two were delivered, so three GREENs
go into the re-audit without an adversary, which is the largest gap in this
report.
