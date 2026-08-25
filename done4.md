# Work done — round 4 (CEQ v6′), the chosen-sign round

Round 1 archived in `DONE_ARCHIVE_ROUND1.md`. Round 2 in `workdone2.md`. Round 3
in `done3.md`. This covers round 4 and is updated every iteration until the loop
stops.

**Goal:** not the best token predictor — the **next-equilibrium predictor**.
Attention that understands causality and consequences on Turing-grade problems at
the smallest scale, and **the module must survive equal to self-attention or
supersede it.**

---

## Where the round stands

Round 4 attacks **ε**, the signs, because rounds 1–3 attacked `t_p` — the term
count, the scale, the rank — and all three died.

**Two of the round's four mathematical warrants have now been measured
non-operative.** That is the headline, and it was found before either arm was
built, which is what the pre-build gates exist for.

---

## The two warrants that fell

**X₂ — Spencer's bound is vacuous at this geometry.** The promise: signs may be
*chosen* with `|Σ ε t| ≤ 6√k`. Scaled correctly by `max|t_p|` — the bound assumes
`|t_p| ≤ 1`, so unnormalised it compares nothing — it reads **4.039e-06** against
a measured natural background of **3.000e-07**. **The background is already 7.4×
below the bound.** Nothing is pressing against it.

And optimal sign choice, computed **exactly** by enumerating all 256 patterns at
k=8 rather than approximated, buys only **18%** (3.000136e-07 → 2.458163e-07).
The reason:

```
mean max|t| / Σ|t|   0.5068 (harness)   0.5684 (unit)
equal terms would be 0.1429
```

**One term carries ~51% of the background mass.** You cannot cancel a term that
outweighs all the others combined — and that domination is extreme-order-statistic
selection doing exactly what round 2 identified.

**X₃ — FKG does not bind.** The claim: selection-coupled signs *must* align. On
identical draws, bound `torch.equal(q,q')` and `torch.equal(k,k')` in-process:

| λ | s | E\|Σε\| | P(+) | pair corr |
|---|---|---|---|---|
| 0.10 | 512 | 7.997000 | 0.9998 | **0.999250** |
| 1.00 | 512 | 1.829000 | **0.4968** | **−0.039893** |

The alignment I measured at round 3 and the contract attributed to FKG **was λ,
not a correlation inequality.** At λ=1.00 the selected signs are balanced and
very slightly anti-correlated.

---

## What was built

**X₄ — the valuation instrument (GREEN).** Building it found a defect nobody had
named. The shipped statistic is `lo*hi < 0 and min(|lo|,|hi|) > floor`, and it
has **two** independent failures:

1. the floor discards true flips — a composed arm reads 0.386719 at `floor=0` and
   0.000000 floored, 100% gone at median |grad| 2.8e-32;
2. **`lo*hi` multiplies, and the product underflows when neither factor does.**
   `np.float32(1e-30) * np.float32(-1e-30)` = **`-0.000e+00`, exactly zero**, both
   operands normal and eight orders above float32's smallest normal. `lo*hi < 0`
   is `False`: a genuine flip counted as no-flip. **Deleting the floor does not
   fix this. Only not multiplying does.**

Calibration: **4/4 exact** against the published table at its own floor, so the
new instrument moves no historical number. That check surfaced a fact about our
most trusted gate — `sign_flip_rate` defaults to `floor=1e-06`, so the
calibration table is itself a floored reading, discarding **57.1%** of `sgate h1`
and **0.0%** of two others. A per-case distortion nobody had recorded.

**G7 — both arms cleared the pre-build event-change gate.** Arm B changes the
event in **20.93%** of draws (harness) and **14.43%** (unit), far above the 1%
threshold, so it is not a no-op the way R5 and R8 were. Arm A's birth gate
`|D−D| = v−1` holds exactly at v = 7, 13, 21, 31, 57.

---

## Two errors of mine, both caught within one iteration

**Arm A's theorem described the wrong object.** The coverage theorem is about
**differences**; a causal two-hop path composes **sums**, because both hops point
the same way. Measured: `|D−D| = 56/56` exactly, `|D+D| = 36/57`, and the probe's
own offset was in neither, so the pair was unreachable under *every* schedule —
which is why all three rows read severance 1.0000 identically. **The uniformity
across schedules is what exposed it.** Eighth appearance of "correct statement,
wrong object", committed one iteration after recording the seventh.

The redirect: the object needed is an **additive basis of order 2** (`D+D ⊇ ℤ_v`).
Sidon sets are exactly backwards — they minimise sum collisions where coverage
needs them maximised.

**And `j = s//4` was itself unsound**, pinning the offset at `3s/4`, unreachable
for any bounded-offset schedule at low hop count. `carpet_probe.py:24` already
demanded uniform-at-random placement for `c`; it was never applied to `j`.

---

## G1 fired, on round 3's proudest result

[arXiv 2606.28560](https://arxiv.org/abs/2606.28560) compares a **"coprime
(anti-gridding) reassignment"**. Anti-gridding is exactly the mechanism round 3
measured — power-of-two lattices grid, co-prime breaks the grid, severance
0.5745 → 0.1277. **Round 3's co-prime repair is prior art**, and the idea
originates earlier still, in dilated CNNs: [arXiv 1702.08502](https://arxiv.org/abs/1702.08502),
verbatim — *"the dilation rate within a group should not have a common factor
relationship (like 2,4,8, etc.), otherwise the gridding problem will still hold
for the top layer."*

Not occupied: difference sets and Sidon sets are **not mentioned** there and were
**not found** in search. Additive bases have **not yet been fetched** — that G1
is still owed and the cell may not be free.

Also established by fetch rather than assertion: kernel herding produces a
*"super-sample"* and its descendants *"select an m-element subset"* — both are
**sample** selection. Arm B's cell is **sign** assignment inside the operator.

---

## The journal, and an instrument that was sampling

A full census of all 37 journalled units found **13 drifting — 35% — across all
three cells**. The differing fields are *only ever* `sigma` (13/13) and `term`
(11/13). **`rate`, `k` and `n` are bit-identical at all 37**, drifts are
~1e-9–1e-10 relative and **bidirectional** — accumulation order, not semantics.
Every published M2 number derives from `rate`, so **the verdict is untouched**.

No thread count repairs it: `s1024/b0` reproduces at 1 and 4 only, `s128` at
**none** of 1/2/4/8/16/20/24. `JOURNAL_THREADS = 2` cannot be fixed by choosing a
better constant, because among those tried none exists.

**Why it hid for 45+ iterations:** `check_replay` reads one unit per run. At
24/37 clean, any iteration had a **65% chance of passing while a third of the
journal was drifted.** It was a sampling instrument and nothing said so.

That failure is worth separating from the nine structure instruments. **It never
gave a false reading** — it read one unit correctly and was taken as a statement
about the journal. The repair narrows the assertion to `rate/k/n`, reports
`sigma/term` as advisory **in the check's own output line**, states the sampling
there, and adds a second must-fire control requiring that an advisory-only
difference is ignored — without which the narrowed check would still be claiming
what it no longer verifies.

---

## What is alive

**Routing beats softmax, and it survived every correction.** At n_train=8192,
4769 params in every arm, softmax measured first:

```
softmax          0.877168   [0.830455, 0.924226]
pivot_unsigned   0.747528   [0.696849, 0.797716]   DISJOINT
```

`pivot_unsigned` carries **no signed content** — plain softmax attention
differing from the baseline *only* by routing hop-2 through 8 content-selected
pivots. It is the one positive result in four rounds, and it is unsigned.

---

---

## THE LEAP — Dr House, 85 seconds, and it reframes rather than patches

Released because two of this round's four warrants had been **measured**
non-operative while eight iterations went to instrument repair — an innovation
gap, not an evidence gap.

**His verdict: the frame is the mistake, and our own numbers already said so.**

`flip(s)` lives in an **additive** world — consequence as one term outshouting an
unnormalised sum, an `O(k^{-1/2})` event that four rounds tried to inflate to
`O(1)`. Attention lives in a **competitive** world: softmax is a normalised
budget on a simplex, so a third token `c` hurts `j` by **displacing** it, not
negating it — and displacement is `O(1)` at every `k`, because **the simplex does
not grow**.

```
D(s) = E_i[ TV( A_i(· | c present),  A_i(· | c masked) ) ]

l_{i→j} = q_i·k_j + Σ_c A_ic (u_c·k_j)
```

Consequence measured as **reallocation of i's read**, never as the sign of a
scalar sum. `pivot_unsigned` stays untouched; `c` enters only the logits. Helps =
`c` opens `j`'s route. Hurts = `c` hands `j`'s mass to a competitor. **Zero-sum
on the simplex IS the sign.** And a fixed point of a competition is an
**equilibrium**, not a token — the "next-equilibrium predictor" bar falls out
rather than being bolted on.

**Why it is worth binding: it turns four rounds of dead numbers into its own
evidence.**

| our measured number | its status in the additive frame | its status here |
|---|---|---|
| `pivot_unsigned` beats softmax, **zero signed content**, CIs disjoint | an awkward survivor | **the frame itself, already measured** |
| signed − unsigned CI **[−0.0458, +0.0063]** includes zero | a failure | its **prediction** — if consequence is displacement, ε on values is an empty channel, measured empty |
| Spencer slack **7.4×**; FKG non-binding (0.9998 → 0.4968) | two dead warrants | irrelevant — `D(s)` never touches ε |
| `max\|t\|/Σ\|t\| = 0.507` | why cancellation could not work | **why TV displacement is large** — concentration is fuel on a simplex |
| magnitude routing **−1.298** | routing died | the **magnitude criterion** died; the live arm selects by content |

**His pre-registered kill, falsifiable both ways.** Same harness,
`k ∈ {8, 32, 128, 512}`, 20000 draws, single-pivot ablation of the top-selected
`c`, measuring **both** quantities: `flip(s)` log-log slope vs `k` must be
**≤ −0.4** (a vanishing event must die on schedule) while `D(s)` slope must be
**≥ −0.1** (scale-free on the simplex). **If `D(s)` slope < −0.3, displacement
dies with flip, the frame offers nothing at global reach, and the answer was "no
leap."**

**It is a HYPOTHESIS and it enters no verdict unbound.** Dr House is exempt from
the RED-test-first rule because five minutes does not fit a test — and that
exemption is exactly why a fellow must bind it, with a failing test, before any
of it is credited.

---

## Currently running

| who | on what |
|---|---|
| **Cameron** | arm A rebuilt on an **additive basis of order 2** (`D+D ⊇ ℤ_v`), with `j` and `c` drawn uniformly at random and the flip read on X₄ — plus `TRAINING.md`: exact parameter counts, what fits a free T4, and a resume-chunked plan, **free tiers only** |
| **Foreman** | root cause of the 13/37 drift — what computes `sigma`/`term` versus `rate`, whether 13 vs 24 tracks a parameter, and what a journal certifies when a third of it cannot be replayed and the producing code cannot be diffed |
| **Wilson + nurses** | the mask cache (the only bitwise-binding optimisation, 1.019×), pinning the thread count in the capability harness — **the one file with published numbers that does not pin** — attacking the 25% operator build, and the owed additive-basis G1 fetch |
| **Dr House** | reported. Displacement frame, above. Awaiting a fellow to bind it. |

---

*Updated through round 4, iteration 8, plus the leap.*
