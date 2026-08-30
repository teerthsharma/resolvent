# R10 — iteration 16: two clauses fail, and the third is a green nobody earned

**Script task.** *"SATURN: C-A split integrity = train/eval same graph family or
graph ∈ x; C-B linear probe within-split must read NRMSE > bar; C-C peak
activation bytes = 4·n·s·d·heads computed in writing vs card bytes."*

| clause | verdict | the number |
|---|---|---|
| **C-A** split integrity | **FAILS** on one of two readable splits | **12 shared graphs**, bit-identical labels, **43.66%** of eval rows |
| **C-B** within-split probe | **FAILS**, 25 of 28 rungs + both pooled splits | median NRMSE **0.7639** against bar 0.9 |
| **C-C** activation bytes | **GREEN, and unearned** | 512.0 MiB vs 8,585,740,288 card bytes — **VRAM never binds** |

**All three were ill-posed as written**, and the amendments are stated as
amendments rather than applied quietly.

## C-A — the reproduction that earned trust destroyed the split

The corpus **declares no train/eval split at all**, so the clause cannot be
evaluated until one is. Both partitions a reader could take were measured rather
than one being chosen silently:

| split | same family | shared graphs | leaked eval rows |
|---|---|---|---|
| by arm (`reproduce` \| `widen`) | yes | **0** | 0 / 5,693 |
| by file (`it14` \| `it15`) | yes | **12** | **4,412 / 10,105 = 43.66%** |

Graph identity is SHA-256 over the sorted edge list; family membership is checked
by **regeneration** — the generator reproduces the edge set bit-for-bit, 40/40 —
not by a resemblance statistic.

**The cause is the interesting part.** Arm `reproduce` *is* it.14's corpus by
design, and that bit-for-bit reproduction (λ₂ delta 0.0) is precisely what made
it.15's second oracle trustworthy. The same fact makes the two files an unusable
split. Trust earned along one axis was spent along another, by the same act.

**And it fails toward the optimistic end** — the opposite of the case the clause
was written for. `impact` died at linear-probe train 4.93e-08 against eval 1.478
with *nothing* leaked and `cos(w0,w1) = -1.077e-08` making transfer impossible.
This leaks everything. Same disease, opposite ends.

The third reading — "graph ∈ x" — is **not answerable**: the corpus declares no
tensorisation. Recorded as an open precondition, not a pass.

## C-B — the probe is reading the recursion, not finding a shortcut

Bar imported **by object identity**: `FAIL_BAR = rips_gate.FAIL_BAR = 0.9`, the
repo's own "doing essentially nothing" bar, already used for this role at
`impact.py:1121`. `demo()` asserts the identity, so a local re-pick fails.

| feature rung | median NRMSE | ≤ 0.9 |
|---|---|---|
| F0 degree | 1.0066 | 0/28 |
| F1 + ball sizes | 1.0060 | 0/28 |
| F2 + boundary geometry | 1.0192 | 0/28 |
| **F3 + `gbar`** | **0.7639** (min 0.5670) | **25/28** |

**Mechanism, exactly:** `u = P_IB g + P_II u`, so `gbar` is *literally the first
Neumann term of the label's own recursion*. The probe is not exploiting an
artifact; it is reading the label's definition.

Bounded the other way too: **0/28 reach `PASS_BAR = 0.5`**. And the verdict turns
entirely on a declaration the corpus does not make — **admissible 28/28 on F0–F2,
inadmissible 25/28 on F3**. The honest report is that the tensorisation is
undeclared, not a pick of whichever feature set gives the preferred answer.

## C-C — the formula prices the wrong tensor, and the resource is wrong too

`4·n·s·d·heads` at batch 8192, s=1024 reads **512.0 MiB**, 6.25% of the card;
25/25 shapes fit. Two reasons that green is not earned:

**1. It under-prices by exactly `s/d = 64`.** The shipped peak is the `[n, s, s]`
operator (`m3_capability.py:144`), i.e. `4·n·s²·heads` = **32,768 MiB** at that
same shape — **four times the whole card**. A clause passing at 512 MiB is
certifying a job that cannot fit.

**2. VRAM has no reachable failure state.** All 25 VRAM rows are
**inapplicable, 0 passed**. Host RSS binds instead: at the round's 2 GiB budget
the host refuses at batch **8,832** with the card at 690 MiB — 9% of free — while
VRAM would not refuse until batch **101,837**, **11.53× later**. `101,837 > 8,832`
is asserted in the code, so the vacuity is executable rather than argued.

This is the it.8 defect again, one clause over: `require(4275, name='capacity-sweep')`
returned FITS on 7,162 MiB of free VRAM for a job that spends none.

## Counts — five categories, kept separate

C-A 4: 1 passed / 2 failed / 0 errored / 0 inapplicable / **1 not_reached**.
C-B 30: 3 / 27 / 0 / 0 / 0. C-C host 20 passed / 5 failed; C-C VRAM **25
inapplicable, 0 passed**. `widen-n128-d4-t0.905` is carried as `not_reached` —
it.15's fifth category — and summed into no pass.

**Every clause was shown able to fail.** C-A: planted leak → `shared_graphs=1`;
planted 32-node ring → `same_graph_family=false`; the clean control does not fire.
C-B: label inside the F1 span reads 5.041e-05 (fires); white noise on the same
features reads 1.6028 (does not). C-C: host and card must-fires both refuse.

## Correction to this record's author

The string `"no .cuda() anywhere in this file"` is at **`m3_capability.py:261`**,
not `:120`. The wrong number was taken from MERCURY's `r10_it8_pricing.py:25`,
written into the `vram_gate.py:37` docstring while composing that module, and then
quoted again in the it.16 brief handed to SATURN — propagated from one agent's
report into a tracked module and out into a third agent's instructions, never once
opened. Both files corrected.

That is the **fifth** inherited claim struck this round, and the rate is now the
finding: five for five, across three sources, by an author who struck others for
the same thing in between. Also corrected: the RSS fit residual is **10.1 MiB**
where it.8 said "within 10" — rounded in the flattering direction.

**One defect I reported was real**, and verified by block structure before sending:
`binding_crossover(host.available_mib, …)` sat inside the `card is not None`
branch with no host guard, and `read_host()` returns `None` without psutil. One
guard added. Robustness, not vacuity — it crashes loudly and cannot produce a
false green.

## Open

- The corpus must **declare a train/eval split** and **declare its tensorisation**
  before C-A and C-B have answers at all.
- Under any reading, **C-B fails on the features that include `gbar`**, and a
  linear probe beating the bar means the task is not measuring what the
  architecture claims.
- `4·n·s·d·heads` should be `4·n·s²·heads` for this harness, or the clause should
  say which tensor it prices.

---

## Working assumptions, declared so the loop does not stall

Phase 1b's gate (it.19) needs C-A…C-F all green, and C-A/C-B cannot be evaluated
until the corpus declares a split and a tensorisation. Blocking would halt Phase
1c entirely, so the round proceeds under two **stated** assumptions:

**1. Split by ARM** (`reproduce | widen`). Measured at **0 shared graphs** over
5,693 eval rows, against 12 shared and 43.66% leaked for the by-file reading, with
a planted-leak must-fire confirming the clause fires. Low risk: of the two
readings, this is the only one that is a split at all.

**2. Geometric features F0–F2, not F3.** **This is not low risk and it is an
architectural decision the author owns.** The reasoning: `gbar` is the first
Neumann term of `u = P_IB g + P_II u`, so an arm that sees boundary values reads
the label's own recursion; the probe then beats the bar at median 0.7639 against
0.9, and the corpus measures nothing about architecture. The only reading under
which this corpus is a valid test is F0–F2.

**If the arm is intended to see boundary values, this corpus must be rebuilt
rather than re-declared, and every downstream S1/S2 result computed under this
assumption is void.** Recorded here, before any downstream work, so the dependency
is visible rather than discovered at the point it invalidates something.
