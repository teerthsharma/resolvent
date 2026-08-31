# R10 — iteration 20: the coherence constant is quoted at a width the code does not run

**Script task.** *"JUPITER: F18–F21 cited or assumed; the coherence line at the
width S1 actually runs."*

## The measurement

`scale/coherence_floor.py:18-30` publishes its numbers at **d = 256, k = 16**.
S1 runs at **d_model = 16, k_pivots = 8** — `scale/m3_capability.py:79` and
`:87`, confirmed executably as a `[512, 64, 16]` key tensor. The `d = 256` in
`scale/foreman_consequence.py` is the **task** width; `:427` passes
`d_model=SF.DMODEL` = 16 alongside it. Census: of 36 numeric `d_model` literals
across 374 `.py` files, **31 are 16 and 0 are 256**.

Monte Carlo, seed 0 declared pre-run, 20,000 trials, mean pairwise max |cos|:

| config | measured | 95% CI | vs the published cell |
|---|---|---|---|
| d=256, k=16 (published) | 0.174795 | [0.174460, 0.175131] | reproduces `coherence_floor.py:29` exactly |
| **d=16, k=8 (shipped)** | **0.547180** | [0.545987, 0.548373] | **3.130×**, CIs disjoint by 0.371 |
| shipped key directions | 0.719784 | [0.698242, 0.741325] | 4.12×, real keys are correlated |

The published cell reproduces. It is the *transfer* to S1 that fails, and it
fails three ways at once, so no single repair covers it.

## Three failures, not one

**Wrong width.** 3.130×, above.

**Wrong quantity.** The scramble control's residual is `|Δ‖k‖|` in key-norm
units; coherence is a dimensionless cosine. A cosine is scale-invariant and the
residual is not, so no unit conversion carries one to the other.
`MATHEMATICS.md:849-853` says **"overlap"**, correctly; the apposition in the
script line is where a cosine became a residual.

**The `k ≤ d` premise fails at two of the three shipped `k`.** `--ks` defaults to
`[8, 32, 128]` in eight modules against `d = 16`. Welch is exactly 0 only at
`k ≤ d`. At the other two it is not:
`welch(16,32) = sqrt(16/496) = 0.179605`, `welch(16,128) = sqrt(112/2032) =
0.234772`. Both checked by hand against the closed form.

## The near-collision that makes this hard to see

`welch(16, 32) = 0.179605` sits **within 2.7%** of the published `0.174795`, and
the two quantities share no argument — one is a worst-case floor over codes, the
other an expectation over random draws. **Five distinct quantities in this repo
read within 15% of 0.175**, including `ceq/diagnose.py:15`'s
`content_conditional_sign_decay` at s=8, which is `0.17480` and unrelated to
either.

That is the surface-proxy shape again, in its numeric form: agreement of two
decimal digits standing in for agreement of subject. Filed against
`R10_MECHANISM.md`.

## A correction to the report that produced this

The report reads *"reproduces the cited 0.1748"*. **0.1748 was never a cited
constant.** The v11.1 amendment's figure is `0.147`
(`.superpowers/sdd/polymorphic-drifting-squirrel/progress.md:451`;
`mu_jl(256,16) = 0.147176`). 0.174795 is this repo's own prior *measurement*,
already published at `coherence_floor.py:29`, which refuted it.

And the existing record understates its own result. `mu_jl` and `mu_pairs` are
both **union bounds** on `E[max]`, so a valid one must sit **above** the
measurement. `mu_jl = 0.147176` sits **below** `0.174795` — it is not a loose
bound, it is **not a bound at all**. The corrected `mu_pairs = 0.193397` sits
10.6% above, which is what union-bound slack costs and is the correct sign.
`progress.md:454`'s *"15.8% low"* prices this as an accuracy error; it is a
validity error.

## Prior state this does not disturb

`AUDIT.md:661` already carries `scale/coherence_floor.py` at **2/11 readings
reproducing at abs=5e-7**. The two that reproduce are the d=256 MC pair
re-measured here. The other nine are untouched by this iteration.

## The smallest amendment

Print the Welch **formula** per row rather than a constant; replace the `d=256`
cell with the measured `d=16, k=8` figures; split the coherence sentence from the
residual sentence, since they are different quantities and only one of them was
ever measured.

## Bound

`scale/r10_it20_coherence.py` — assert-based `demo()` with four
instrument-failure checks: Welch tight at d=2,k=3; an un-normalised sampler must
leave the CI; the matcher reads zero only where it should; and a must-fire that
the two dimensions' CIs are disjoint. `results/r10_it20_coherence.jsonl`, 12
rows. VRAM gate passed at `host_mib=512`; one process, no GPU, no training.

## F18–F21

All four **cited**, none assumed: F18/F19/F20 at `CHECKLIST.md:368/372/375`, all
four at `LOOP_PROMPT_ROUND4_ARCHIVE.md:50/55/60/65`. F21 exists only in the
archive, not in the register.

**The F-numbering collides across three namespaces** — `F0–F3` (features, it.16),
`F1–F8` (it.14 prerequisites), and `F24+` under a *"New fetches"* heading at
`PHASE2_CONTRACT_V_MAIN_4.md:69`, which implies a running series through F23.
**No F9–F23 table exists in this tree** (129 `.md`, 374 `.py` searched). Both
readings are recorded; no subject was invented for the second.

F21 is the one that bears load here — the matcher runs on `batched_select_pivots`,
and its forward-only limit above n=64 does not bite, since every reading in this
iteration is a forward pass.
