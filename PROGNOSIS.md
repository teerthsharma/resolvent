# Prognosis

Written 2026-08-25, loop iteration 21, immediately after a Health Inspector pass
(3/3 mechanical checks clean, 3 claims struck on provenance).

Every number below carries its evidence class — **RUN** (executed here, output
in `results/`), **READ** (file:line), **CITED** (external, resolved), **DERIVED**
(from RUN/READ by stated steps) — and, per the standing rule added in iteration
20, **every agent-produced figure names its source inline every time it appears,
not only at first mention.**

---

## The verdict

**The attention operator this project set out to build does not work, and the
measurement that said otherwise was measuring something the module does not
ship.** On the shipped operator the pre-registered kill fires by a factor of
four, and the property the whole design rests on is *absent* above context 32.

**What survives is not the operator. It is the Lean core as MATHEMATICS, and the
falsification harness** — 27 machine-checked theorems with no `sorry` and no
`sorryAx`, and an instrument discipline that caught seventeen of its own broken
instruments, including the one that invalidated the headline.

**Not the certification.** [RUN, iteration 28] `occupancy_is_exact_inverse` is
stated at `N = n`, and the module truncates at `hops = 2..4` where `‖A^hops‖` is
**0.880500 at s=128** — not zero. **The theorems certify a computation the module
does not perform**, so M5 is RED and the phrase *"machine-checked finite
resolvent"* is overstated wherever it appears. The resolvent is exact at
**hops ≥ n**; the module runs **hops = 2**.

That is a real artifact and it is worth shipping. It is not the artifact the
project set out to make.

---

## What died, and by which number

**M2 — context-stable signed influence at global reach. RED.**
[RUN] On the shipped operator (`sgate`), with `c` drawn from the pivot set to
match the journalled protocol: rates **0.16511 / 0.02732 / 0.00000 / 0.00000**
at s = 8/32/128/512, slope **−1.298** against a pre-registered bar of **−0.3**.
**Zero flips at s ≥ 128**, on 213 and 60 usable draws. Pivot routing makes it
*worse*, not better: dense reads −1.088.

**THE RED IS NOW PERMANENT, AND IT WAS EARNED THE HARD WAY [RUN, iteration 42].**
The statistic is `lo*hi < 0 AND min(|lo|,|hi|) > floor`. Condition 1 is a sign
change; **condition 2 is a MAGNITUDE GATE at `floor = 1e-6`** — the same floor
this project already convicted once, when the published exponent −1.389 was
withdrawn as a floor artifact. If gradients shrink with `s`, the rate decays even
when the sign structure is untouched.

Tested paired, both floors on identical draws in one pass so the difference is
exact rather than two noisy runs compared:

| | floor = 1e-6 | floor = 0 |
|---|---|---|
| slope | −1.0938 | **−1.1150** (R² 0.9350) |

**The kill fires with the gate removed** — still nearly four times past the
−0.3 bar — and removing it makes the slope **steeper**, not flatter. That is the
opposite of what a rescue needs.

**But the floor does real work, and its effect grows with context:** it discards
**6.8% → 52.6% → 100%** of genuine sign changes at s = 8 / 32 / 128. **At s=128
the published `0.00000` does not mean "no sign changes" — there were two, and the
floor discarded both.** The verdict is unchanged; the way the number reads
overstates it, and that is a defect in the reporting, not in the finding.

**And the exponent is better supported than it looks.** The tail zeros rest on
213 and **60** draws — at s=512 the CP upper bound is **0.05963**, so the true
rate could exceed the well-measured s=32 rate. Yet the slope through the two
*solid* points alone is `log10(0.02732/0.16511) / log10(4)` = **−1.2977** against
the published **−1.2980**. **The weak zeros move it by 0.0003.** A tail
re-measurement cannot rescue M2 and is not worth buying.

**Correction to the dense control, [RUN] and it makes the separation
stronger.** `s=8` was **degenerate**: `select_pivots` excludes `i` and `j`,
leaving six indices, so |P| = **6, not 8**. Routing restricted nothing there —
`A[:,P]A[P,:]` equals `A@A` to **1.86e-09**, under the probe's own floor, and
**both s=8 cells read k=101, n=4096**: one arm reported twice under two names.
It was the **leftmost point of both fits**. Dropping it moves the dense slope
from −0.746 to **−1.009**. Replacement is s=16, the smallest size where |P| = 8
is a proper subset; routing bites when **s > 13.3** at k=8.

This costs the claim nothing and costs the wrong number everything — **and it
does not rescue M2.** A stronger separation between two arms of an operator that
ships nowhere is still a fact about an operator that ships nowhere.

**M2′ — the four-route replacement. RED.**
[CITED — Foreman] The twin test runs each route's own criterion on `A` and on
`|A|` (magnitudes bit-identical, signs stripped): determined fraction
**0.8555 → 1.0000**; balanced ambiguity **0.0410 → 0.0000**; off-support
influence **0.000e+00 → 0.000e+00**; slope(term/σ) **−0.04/−0.92 →
−0.046/−0.930**. **Every criterion is met at least as well by an operator with
no signs at all.** All four routes measure a sign-blind scalar.

**The sign branch is closed by argument, not by a failed test.**
[CITED — Chase] The readout is `A[i,j] + Σ_{p∈P} A[i,p]A[p,j]`, and `c` enters
**exactly one term**. So "does `c` flip the sign" is exactly "can `|t_c|` beat
`|`the rest`|`" — a magnitude comparison. **Sign structure can forbid the flip;
it can never protect it.** That retires R2 and arsenal item C1 as a branch.

**Path coherence — the named replacement. RED, and backwards.**
[CITED — Chase] Dense tgate at s=2048 sits **7.8×** above its null, deltanet
**13.4×**, but the **pivot bundle excess is only 1.14×**. Routing does not
*create* coherence; it cuts N and *reduces* it.

---

## Instrument #17 — the one that matters

**Every M2 and S2 headline was measured on `tgate`, which the module does not
ship.**

[RUN] `_causal_tgate_operator` lives in `ceq/bench.py`, five `scale/` probes and
three test files. The shipped module uses `ceq_operator`
(`ceq/attention.py:151,249`) and `sgate_operator` (`modeling_ceq.py:297,404`).

[RUN] And the arm names hid it. `scale/pivot_probe.py::ARMS` **never contains
the string "tgate"** — the arms are `pivot_signed` and `dense_signed`, and
`build_arm` maps **both** to `_causal_tgate_operator`:

    deltanet        -> _causal_deltanet_operator
    dense_signed    -> _causal_tgate_operator     <- ships NOWHERE
    dense_unsigned  -> _softmax_operator
    pivot_signed    -> _causal_tgate_operator     <- ships NOWHERE
    pivot_unsigned  -> _softmax_operator
    sgate           -> _causal_sgate_operator     <- SHIPS

**The name describes a property — "signed" — not an implementation.** Nothing in
any arm list, any table, or any of this project's own records ever said the
numbers came from an operator that ships nowhere.

**This is a different species from the other sixteen.** Those were broken
measurements, catchable by calibration. **This was a correct measurement of the
wrong object**, and no calibration could have caught it — the instrument worked
perfectly throughout. What was missing was a bind between *measured* and
*shipped*, and the repo already had that pattern for Lean (`.tril(-1)`
grep-binding a theorem hypothesis to the shipped tensor) without ever applying it
to the operator. It now exists:
`tests/loop/test_measured_operator_is_shipped.py`, 14 passed, calibrated at both
ends.

---

## The instrument taxonomy — the most transferable thing here

Eighteen instruments in this project were internally consistent and externally
wrong. Sorted by what they compared, they fall into two groups with **completely
different failure rates**, and the split is the finding:

| | instruments | times they gave a false reading |
|---|---|---|
| compared **structure** (regex, slices, substrings, exit codes) | 7 | **7** |
| compared **values** (numbers against numbers) | 2 | **0** |

The seven structure failures: the LOCK slice boundary; the LOCK-line scraper
matching prose; the provenance audit missing a line break; the ARMS-DISTINCT
dispatch slice broken by a refactor; a line-scoped strike-marker check whose
markers sat one line away; a substring search for the count `809` that matched
`0.038097`; and `lake build | tail; echo $?`, which reports **tail's** exit
status, not the build's.

The two value instruments — the calibration gate and `scale/bucket.py`'s bitwise
replay — have never once been wrong.

**The lesson is not "write better regexes."** It is that a check comparing
*structure* is checking a proxy, and proxies drift when the thing around them is
reformatted, refactored, or piped. A check comparing *values* has nothing to
drift. Every bind added late in this project was built value-first for that
reason, and none has failed.

**Two corollaries, both learned by being wrong:**

- **A presence check cannot prove absence.** Iteration 34 declared the documents
  consistent on the strength of a script confirming each one *contained* the
  verdict string. Two documents were carrying **struck** numbers the script had
  no way to see. Absence is now mechanised as a value walk over the shipped
  objects, plus a paragraph-scoped text layer that is explicitly labelled weaker.
- **A presence check also fails when you guess the wording.** Iteration 39
  searched for "random init / untrained / random-init", found nothing, and
  concluded the repository had never recorded that its measurements use untrained
  weights. **It had** — the last line of the Open list below, in the words
  "random projections". The claim of novelty was wrong; the gap was real.

## The fabricated number, and why fourteen passing tests did not catch it

`‖A^hops‖ = 1.471448` was published in this document, in `README.md`, in
`MODEL_CARD.md` and in a test docstring. **It is not reproducible.** A sweep of
1,800 settings — dimension, seed, generator layout, `rho`, `lam`, `hops` —
produced it exactly **zero** times. The measured value is **0.880500**.

It survived because the test beside it asserted `got > 1e-3` — **an
inequality**. Both the true and the fabricated value satisfy that, so fourteen
tests passed around a false docstring. It also sat near a real reading
(`(64, 2) = 1.476635`), so it never looked wrong.

**A test that pins an inequality cannot protect an exact number quoted from it.**
The repair was not the correction but the pin: every measured `(s, hops)` value
is now asserted at `abs=5e-7`, and substituting the fabricated value back fails
exactly one test.

---

## What survives audit

**The Lean core, as mathematics.** [RUN] 27 theorems, `lake build CEQ` exit 0,
zero `sorry`, no theorem depending on `sorryAx`. It is indifferent to which
operator is instantiated, so instrument #17 does not touch it.

**M5 IS RED, AND IT LIMITS WHAT THIS SECTION MAY CLAIM.** [RUN, iteration 28]
`A^n = 0` is confirmed empirically — exactly `0.000000e+00` at s = 16/64/128/512,
matching `pow_card_eq_zero` to the bit. But the shipped truncation leaves
`‖A^2‖ = 0.880500` at s=128 and `1.292741` at s=512, so
`occupancy_is_exact_inverse`'s `N = n` hypothesis **is violated by the tensor
that ships**. And no bound replaces it: `truncation_bound` refuses at the
shipped `rho = 1.5`, where the geometric expression is **−6.75**, a negative
bound. **At shipping settings the truncation is unbounded.**

**This is the fourth appearance of one shape** — a correct statement about an
object other than the one that ships, after instrument #17 (`tgate`), M4's kill
(deleted content), and M2's clause 2 (structural zero). Iteration 18's bind
covers **operators**; **nothing yet binds theorem hypotheses to shipped
settings.**

The theorems themselves, unaffected:

- `Nilpotent.pow_card_eq_zero` — strictly-lower-triangular `A` over any
  `CommRing` has `A^n = 0`, with **no sign hypothesis and no magnitude
  hypothesis**.
- `Occupancy.occupancy_eq_inverse_of_nilpotent` — the truncated sum **is** the
  two-sided inverse of `(I − A)`.
- `Refcount.floor_add_orbits` — caustic Theorem 1's indistinguishability floor
  equals foliation's refcount, `n − m = Σ_plaques (refcount − 1)`. Both halves
  are the author's own prior work; this is the sentence that identifies them.

**The harness.** Append-only journals; PID locks; **bitwise replay as a
determinism audit** (three independent journals — `m2`, `s2`, `r2` — each
reproduce from a fresh process); pre-registered kills; theorem-backed
calibration anchors that return *exactly* zero; and the rule that **a kill which
cannot fire is a defect**.

That last rule earned itself repeatedly. [RUN] R2's kill was unreachable — the
determined fraction has a structural floor of **2/k = 0.25** and the kill said
"≈ 0". [RUN] R4's `θ*` is a **half-line, not a point** — slope is identically
zero for every θ ≥ 0.5 out to 1.5, max |slope| **0.055**. [RUN] M2's own clause
2 was zero by construction, and the verdict code mapped its NaN to GREEN.

**The M3 bar is reachable.** [RUN] softmax **1.725106 → 1.304590** for 4× the
data, train/eval gap closing 0.532 → 0.733 exactly as an overfitting diagnosis
predicts. [CITED — Cameron] a learnability control with routing free reaches
held-out **0.0071**. So M3 is not a termination clause; **any arm failure is
routing, not budget.**

---

## What was never done

- **ARC-AGI has never been scored. No Turing-style evaluation file exists.**
- **Nothing has trained above 3.65M parameters** against a 300M gate.
- **The whole test suite has never completed a run** — nine attempts, three
  agents, one 1-hour monitor. 829 tests collect. **No total pass/fail count
  exists for this repository, and none should be quoted.**
- The one capability comparison ever run at matched parameters went **against**
  the operator: COGS-gen softmax **0.0293** (15/512) against **0.0000** (0/512)
  at 3,652,096 parameters in both arms, one-sided Fisher **p = 2.7502788939e-05**,
  and behind **in-distribution** too (0.9258 against 0.7734).

---

## Novelty

**None claimed.** Every novelty claim was withdrawn against prior art this
project found itself: SimA (2206.08898), **Signed Dual Attention (2606.04833),
which is this module's own `sgate` matrix**, DeltaNet (2406.06484), ParaFormer
(2512.14619), SignGT (2310.11025), Cog Attention (2411.07176), RetNet
(2307.08621), Star-Transformer (1902.09113, which owns pivot/relay routing from
2019).

**The per-route sweep fires too, and it was run 26 iterations late.**
`CONTRACT.md` requires G1 as *step 0*; the per-route half ran at iteration 26.

| route | verdict |
|---|---|
| **R3** non-Archimedean / max-plus | **OCCUPIED.** [CITED] *Tropical Attention*, arXiv:2505.17190 (22 May 2025) — *"operates natively in the max-plus semiring of tropical geometry"*, maps Euclidean → tropical → back, leaving subsequent blocks unchanged. **That is R3.** |
| R1 group testing / d-disjunct | **not found** — the surrounding space is crowded (NSA 2502.11089, MoBA, TidalDecode 2410.05076) |
| R4 Dyson hierarchical / RG | **not found in ML** — long-established in statistical physics, no transformer intersection surfaced |
| R2 sign-solvability | swept Round 1 — Brualdi–Shader, CUP 1995, no ML application found |

**And it undermines R3's premise, not merely its novelty.** [CITED] the same
literature records that *in the β→∞ regime, self-attention operates in the
tropical semiring, and the tropical limit of softmax attention is a tropical
matrix product.* R3 existed to break the **Archimedean sum** hypothesis — but if
tropical attention is softmax's own limiting regime, **it is not an escape from
softmax at all.**

R1 and R4 are recorded as **not found**, not as *unoccupied*: absence in one
sweep is weaker evidence than presence, and six novelty claims here have already
died on prior art an earlier sweep missed.

**No work was wasted — because the cost order put R3 last and all four routes
died before reaching it. That is luck, not process.** Under order 3 → 1 → 4 → 2
this project would have built a published architecture and found out afterwards.

And the strongest framing is measured false. [RUN] At s=8 the **unsigned** arm
flips signs at **0.1025** against the signed arm's **0.0264** — softmax flips
**four times more often** at short range before dying. "Softmax cannot do this"
is false, and [READ] one GELU between two softmax layers restores the property
outright (`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`).

---

## Recommendation

**Ship the negative result, the Lean core and the harness. Do not ship an
attention claim.**

A negative result with a reproducing instrument and a machine-checked core is
publishable and useful, and almost nobody ships one. The three things this
repository can defend are: a formal core that does not depend on the operator; a
falsification harness that caught seventeen of its own instruments including the
one that invalidated its headline; and a clean, documented account of why a
plausible design does not work.

**Both blockers this document originally named are now closed.**

[RUN, iteration 22] **M4's kill is calibrated at both ends.** The defect was
real — `settle_evicted` reads `x[keep]` and nothing else, while the perturbed
token came from `lowest_salience_token(..., exclude=keep)`, so `y[keep] ==
x[keep]` bitwise and the headline `0.000000e+00` was zero *by arithmetic*. A
must-fire arm now perturbs a token **inside** `keep` and requires movement:
**8/8 draws**. The crushed-token zero is re-labelled **structural**, and gating
is shown to move in **8/8 draws under both placements**, which is a fact about
the denominator rather than evidence eviction is exact. M4's claim survives; the
like-for-like framing of `0.000000e+00` against `2.154868e-05` does not.

[RUN, iteration 23] **Both NaN→GREEN paths are deleted.** Proved by execution
first: `a = −0.050, b = NaN` printed **`M2 = GREEN`**, and a NaN slope printed
*"signedness is load-bearing… the unoccupied cell"* — this project's strongest
sentence — for an undefined slope. The root cause was **two copies of one rule**:
`_verdict()` had already been repaired, `report()` kept a private copy of the
original, and the tested copy was correct while the copy that ran was not. The
fix is deletion — one verdict path — and it now prints `M2 = VOID — a mandatory
clause is UNEVALUABLE` and refuses.

**The rule both repairs share, and it generalises past them:** *0.0 from an
instrument that CAN move is a result; 0.0 from an instrument that CANNOT is a
tautology.* That one sentence covers M2's clause 2, R2's unreachable 2/k floor,
R4's half-line θ\*, and M4 — four defects across four routes, one shape.

---

## Open

- The **−1.298** slope is fitted on **two nonzero points** (sgate reads exactly
  0.00000 at s=128 and s=512). The **verdict** is robust — 0.16511 → 0.00000
  over a 16× growth, against a bar of "< −0.3" — but **the exponent is not a
  measurement.** [RUN, iteration 42] **This is now quantified rather than
  hedged:** the two solid points alone give **−1.2977** against the published
  **−1.2980**, so the zeros contribute 0.0003 — and the kill survives at
  `floor = 0` (**−1.1150**), which is the mechanism that would have explained it
  away.
- Whether the M3 ordering survives at adequate budget is unknown. [CITED —
  Cameron] at s=128, d=42: softmax **1.7238** [1.5078, 1.9957], pivot_signed
  **1.1846** [1.0921, 1.2930], pivot_unsigned **1.6634** [1.4858, 1.8870] —
  pivot_signed beats softmax with non-overlapping CIs and pivot_unsigned tracks
  softmax, **but every arm is above the 1.0 bar.** An ordering below a failed bar
  is not a result.
- Everything is measured on **random projections**, not trained checkpoints.
  **RESOLVED as far as it can be without spending a run [RUN, iteration 42].**
  The worry is that the operator's dilution is an artifact of initialization. It
  is not testable directly without training, but the *instrument* was tested
  instead, and it survived both suspicions: the kill fires at `floor = 0`, and
  the same instrument reads **−0.034 for `tgate`** against **−1.298 for
  `sgate`**, so it is not returning a constant regardless of input.
  `M2_TRAINED_PREREGISTERED_READING.md`, written **before** that test, fixed the
  consequence in advance — *"if the instrument survives, a trained run cannot
  overturn M2 no matter what it reads."* It survived.
  **And the bar makes the point moot anyway:** the free-T4 shape is 25.7M
  parameters, **8.6% of the 300M gate** this project is held to. A trained
  reading there could not settle the question even if it were run.
