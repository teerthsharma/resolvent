# SATURN (WATSON) — iteration 2 report

Unit: build the vector consequence corpus. Role unchanged — instrument-keeper.
No claim below entered a verdict except through a test that was RED first.

**Status: DONE_WITH_CONCERNS.**

Worktree fast-forwarded `4f619f2 → 1edc963` (`git merge --ff-only
feat/r9-causal-consequence`) before any edit. Mercury's and Venus's work in
`scale/negation_scope.py` is in the tree my changes sit on.

---

## 1. What was built

`c1_propagate_t{1,2,8,32}` in `scale/negation_scope.py`, registered in
`M3_TASKS` and `E_T_STAR`, routed in `e_hop_reading`, with a 23-test admission
bundle at `tests/cameron/test_c1_propagate_registration.py`.

The label is `[n, s − t*]` — one consequence value per position, no scalar
index anywhere:

    y_p = sum_{h=1..t*}  ( prod_{i=p-h+1..p} a_i ) * b_{p-h}

`a` Rademacher, `b` Gaussian, both the chain family's own objects.

### Why not the obvious label — this is the load-bearing design decision

The cheapest vector label is the chain family's prefix scan `z_p`, the
intermediate state of `equilibrium_oracle`'s loop, at zero new code. Venus's
§2b says it is worthless and she is right; I reproduced her measurement
independently before building anything else:

| `t*` | `z_p == b_p` **bitwise** | as fraction |
|---|---|---|
| 1 | 63 | 63/64 |
| 2 | 62 | 62/64 |
| 8 | 56 | 56/64 |
| 32 | 32 | 32/64 |

`make_equilibrium_batch` sets `a[:, :head+1] = 0.0`, so below the head the scan
degenerates to `z_p = b_p`, and `b` **is** the input channel `CH_FLIP`. That is
the exact defect `b[:, s-1] = 0.0` was introduced to remove
(`scale/negation_scope.py:376-387`), re-opened at every position that was never
zeroed. Its zero-hop reading is `0.125709 / 0.214220 / 0.602482 / 0.944499` —
below the bar at all four rungs, which is `INSTRUMENT BROKEN` under
`m3_capability.py`'s RED gate.

C1 fixes the **depth** instead of the position. Dropping `h = 0` means `b_p`
never enters `y_p` — `b[:, s-1] = 0.0` applied at every position, and applied to
the *label* rather than the input, so the drivers keep full variance and no
position's label is constant.

### The regime, chosen deliberately, as asked

**NILPOTENT / terminating.** The sum has `t*` terms and stops. `t*` is an exact
hop count, not a tolerance — the coordinator's lean, and it is also the chain
family's own documented choice for the same reason. The contraction alternative
would have forced the relative form `got <= zeroth * L**k + 1e-9`
(`test_m3_etasks.py:208`), because NRMSE normalises by `std(y)` and not by
initial error. That form is not used here because it is not the regime this
corpus is in. Termination is shown, not asserted: `k = t* + 1` is still
**bitwise** equal to `y`, and `e_ladder_ks` prints that rung precisely so a
contraction and a nilpotent operator look different there.

### The channel decision, as asked

**Reuse. No new channel, `d_model` unchanged at 16.** The ladder is full
(`CH_FLIP 0` … `CH_DOC 15`, plus `CH_HET` from iteration 1), and C1 needs
exactly the two objects the chain family already owns: `CH_DRIVE` for the
coefficients, `CH_FLIP` for the drivers. Raising `d_model` for one family would
have moved every other task's tensor for nothing. `t*` rides in `f` (the chain
head, `s−1−t*`), the same round-trip `make_equilibrium_batch` uses, so the dial
reaches the oracle through a signature every consumer already passes and no
registry entry can pair a builder with another rung's oracle. `_check_dial`
raises on an `f` that does not decode — `_e4prime_marks`'s precedent.

### The band, and why it is `s − t*` and not `s`

A strictly causal operator raised to the `t*` power vanishes on the first `t*`
coordinates — always. Causality and uniform depth cannot both hold on all `s`
positions. An `[n, s]` label would therefore carry `t*` entries of `sd 0`, which
is the fourteenth strike wearing a new shape (MISTAKES.md V-8). `[n, s − t*]` is
still vector-valued and has no degenerate position. That is the one place I did
not follow the dispatch's literal `[n, s]`, and the reason is structural.

---

## 2. Claim ledger

| # | Claim | Class | check: |
|---|---|---|---|
| 1 | Nothing named `make_propagate_batch` existed | RUN | RED run: **19 failed, 0 passed in 0.40s** |
| 2 | Label is `[n, s−t*]`, vector-valued | RUN | `(2048,63) (2048,62) (2048,56) (2048,32)` |
| 3 | Every position is equally deep: `Var(y_p) = t*` | RUN + DERIVED | `t*` independent unit-variance terms with unit-modulus weights; drawn max relative deviation `0.063939 / 0.074059 / 0.071635 / 0.067238` |
| 4 | No position is degenerate | RUN | per-position `sd` min `0.968676 / 1.365359 / 2.761187 / 5.463367`, all finite |
| 5 | `R_0` is the empty sum, bitwise zero | RUN | `torch.equal(r0, zeros)` at all four rungs |
| 6 | `k=0` is the mean predictor, **identically** | RUN + DERIVED | `nrmse(0,y) = sqrt(1 + mean(y)²/var(y)) ≥ 1.0` is an identity; drawn `1.000000357 / 1.000010754 / 1.000007066 / 1.000009670` |
| 7 | Truncation law `sqrt((t*−k)/t*)`, no fitted constant | RUN + DERIVED | worst `\|NRMSE − closed\|` over the printed ladder `0.000000 / 0.000889 / 0.001519 / 0.007071`; across 8 seeds and every `k`, worst `0.011145` |
| 8 | Monotone tightening, `k=1` bounded away, exact at budget | RUN | strict `<` asserted at every rung; `k=1` reads `0.706218 / 0.935988 / 0.984570`; `k=t*` and `k=t*+1` both bitwise equal to `y` |
| 9 | One function serves ladder and label | RUN | `torch.equal(propagate_oracle(x,f,p), propagate_hop_reading(x,t*,t_star=t*))` |
| 10 | The do()-bit moves exactly the right positions | RUN | moved label columns `1 / 2 / 8 / 32`, equal to `t*`; every other column **bitwise** unchanged |
| 11 | Flipper dependence `2√t*/(s−t*)`, closed form | RUN + DERIVED | drawn `0.031795 / 0.046178 / 0.099031 / 0.367110` vs closed `0.031746 / 0.045620 / 0.101015 / 0.353553`; worst deviation over 4 rungs × 8 seeds `0.013556` |
| 12 | The registration calibrates through its own entry | RUN | `payload_only 1.060161`, `oracle 0.000000`, `flipper_dependence 0.101048`, `trained_two_feature 0.075476`, BAR CALIBRATED |
| 13 | The bundle rejects the construction it was written against | RUN | prefix-scan zero-hop reads `0.125709 / 0.214220 / 0.602482 / 0.944499`, all `< 1.0` → REJECTED at all four rungs |
| 14 | The `calibrate_bar` guard is a no-op for scalar labels | RUN | `payload_only` bitwise identical before/after: `1.366458892213088` (negation_scope), `1.050147990558408` (e3_t8) |
| 15 | CLI `--task` choices track the registry automatically | READ | `m3_quintuple.py:617` and `m3_capability.py:247` both `choices=list(M3_TASKS)` |
| 16 | `run_arm` cannot yet train on a vector label | RUN | `RuntimeError: The size of tensor a (64) must match the size of tensor b (56)` — fails loudly, not silently |

**Adversarial pass.** The strongest claim is 3 (uniform depth), because
everything about dilution rests on it. Would the bundle pass with the
construction replaced by the wrong one? Claim 13 answers that by running the
assertions against the prefix scan — and it exposes a real weakness in my own
bundle worth stating: **the equal-variance clause alone does NOT reject the
prefix scan at `t* = 1`** (it reads `0.0638`, inside my `0.10` tolerance),
because 63 of its 64 positions are the same unit-variance copy — uniformly
*uninformative* rather than uniformly *deep*. The zero-hop clause rejects it at
all four. Two clauses, failing differently; a bundle resting on either alone
would have admitted the worst rung of the worst construction.

---

## 3. Venus's prediction, recomputed against the label actually built

`R9_IRENE_PREDICTION.md` §6 makes this an obligation on whoever builds the
corpus: *"PREDICTION 1's numbers must be recomputed from the same formula
against the label actually built, before this file is scored."* Done. Her
formula is `predicted margin = scalar margin × sqrt(w)`, `w` the share of total
label variance at the one position where the arms differ. C1's per-position
variance is uniform at `t*`, so `w = 1/(s − t*)` exactly.

| `t*` | positions | `w` | `sqrt(w)` | predicted `settled − softmax` | predicted `twin − softmax` |
|---|---|---|---|---|---|
| 1 | 63 | 0.015873 | 0.125988 | −0.024827 | −0.020288 |
| 2 | 62 | 0.016129 | 0.127000 | −0.007715 | −0.005528 |
| 8 | 56 | 0.017857 | 0.133631 | +0.004809 | +0.005381 |
| 32 | 32 | 0.031250 | 0.176777 | +0.010545 | +0.007711 |

**All eight are still below the pre-registered resolution `0.027260`, and at
`t* = 8` and `t* = 32` they are SMALLER than the numbers she filed** (+0.010229
→ +0.004809; +0.013881 → +0.010545). Removing the free-copy positions does not
rescue the contract's prediction. It is class `DERIVED`: her formula applied to
C1's measured variances, and it inherits her §2a read.

**That is the point of building the corpus this way, and it is worth stating
plainly.** On the prefix-scan label a null reading is uninterpretable — it could
be the label (63 of 64 positions a free copy) or the arm (one row of 64
architecturally distinct). C1 makes the label uniform by construction, so **a
null is the arm**, and Venus's own falsifier 5 — writing `_alpha` at every
position rather than only `s-1` — becomes a clean one-line test instead of a
confound. This corpus does not decide between the two predictions. It removes
§2b and §2c so that §2a can be decided on its own, which is what "let her be
shown right or wrong rather than hide the answer" required. The table above is
recorded in the bundle's module docstring so it lives in the repo and not only
in this report.

---

## 4. A weakness recorded rather than papered over

A vector label divides one driver's influence across the whole band, so C1's
exact flipper dependence is `2√t*/(s−t*)` — `0.031746` at `t* = 1` and
`0.045620` at `t* = 2`, both **smaller than `bar_verdict`'s default
`flipper_tol = 0.05`**, which was calibrated for `y = payload × sign` whose
ratio is 2.0. At those two rungs the shipped band therefore accepts `0.0`, a
flipper-blind label. That is a gate satisfied by construction (MISTAKES.md
V-10), so `test_the_shipped_flipper_band_is_too_loose_for_this_family` asserts
it — including that the loose set is exactly `[1, 2]` — rather than leaving it
to be rediscovered as news.

The tolerance this family needs is `0.02`, and both ends are measured: it
accepts the drawn spread (worst `0.013556` over 4 rungs × 8 seeds at n=2048) and
rejects `0.0` at every rung (nearest closed value `0.031746`). The anti-vacuity
duty is carried by the movement test, which asserts the exact **support** of the
perturbation bitwise — a strictly stronger check than any ratio band, and it
doubles as the causality check (`BASE_PROMPT`: perturb `j`, assert `i < j`
bitwise unchanged).

---

## 5. Blast radius

One shared function changed: `calibrate_bar`'s `payload_only` clause. It
compared a `[n]` payload against the label, which raises on an `[n, m]` label
because torch aligns trailing axes. The guard broadcasts the payload across the
label's trailing axes — the flipper-blind predictor is "the same payload at
every position", so the clause's meaning is unchanged, and for a scalar label
the loop does not run and `expand_as` is a no-op. Verified bitwise (claim 14).

Everything else is pure addition: a new family, four `M3_TASKS` entries, four
`E_T_STAR` entries, one `e_hop_reading` branch that fires only on C1 names.
`scale/m3_quintuple.py` was not touched. Nothing was trained.

| file | result |
|---|---|
| `tests/cameron/test_c1_propagate_registration.py` | **23 passed** (RED first: 19 failed, 0 passed) |
| `test_m3_etasks.py` + `test_m3_counter_squared.py` + `test_u1_rag_registration.py` + `test_bar_control_sees_the_arms_preprocessing.py` + `test_e4prime_registration.py` + `tests/mercury/test_r9_mechanical_fixes.py` | **106 passed, 0 failed** |
| `tests/chase/test_m3_ladder_task.py` | **14 passed** (the registry/CLI-choices test) |

143 passed, 0 failed. The full suite was never run.

---

## 6. What I could not validate

**No arm has been trained on C1 and I hold no execute seat, so nothing here is
evidence about capability** — the corpus calibrates, and that is all. The
handoff boundary is measured rather than assumed: `run_arm` raises
`RuntimeError: The size of tensor a (64) must match the size of tensor b (56)`
on a C1 batch, because the arm's readout is `[n]` and the label is `[n, 56]`.
It fails loudly rather than silently, but it fails, so C1 is not runnable until
Neptune's arm-side change lands, and I did not verify that the change Venus's
falsifier 5 describes (writing `_alpha` at every position) is in fact a one-line
edit — I read `m3_quintuple.py:302-306` but did not attempt it, since that file
is Neptune's.

The uniform-variance claim is measured at one seed per rung at `n = 2048`
(max relative deviation `0.074`), with the truncation law and flipper band
measured across 8 seeds but the per-position variance not; a per-position
pathology that is stable across seeds at one seed's draw would not have been
caught, though the closed-form derivation says there cannot be one. My
`TOL_VAR_REL = 0.10` is above the drawn spread and far below the prefix scan's
`0.53–0.97`, but claim 13 shows it does not separate them at `t* = 1`, so that
clause is weaker than the other three and I have said so rather than quoting the
bundle as uniformly strong.

The §3 recomputation of PREDICTION 1 is `DERIVED`, not measured: it applies
Venus's formula to C1's variances and inherits her §2a read of
`m3_quintuple.py:302-306` wholesale. I verified that read exists but did not
re-derive whether the pivot term genuinely cannot reach other rows through the
MLP and the residual — if it can, the `sqrt(w)` scaling is wrong in the
contract's favour and my table is too pessimistic. `w = 1/(s−t*)` also assumes
the arms differ at exactly one position; if Neptune changes the arm, the whole
table must be recomputed again and this one is void.

Finally, C1's rungs mirror e3's dials so the two lanes look comparable, but they
are **not** the same task and no margin, NRMSE or row from the scalar ladder may
be carried across in either direction — Venus's row Ω governs, and I have not
built the matched-position control it obliges (the same two arms compared at
position `s-1` only, inside the vector lane, reproducing the scalar ladder's
margins). That control cannot be built until an arm can train on C1 at all.
