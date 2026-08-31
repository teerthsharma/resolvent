# Done

One entry per completed backlog item. Each carries its test name and the measured result,
or the reason it was declined. Declining is a valid outcome and gets an entry.

---

## Round 0 — the differential (pre-loop)

- **Repository scan**, 122 repos → `NOTES.md`.
- **Theory written and attacked** → `THEORY.md`, superseded by `REQUIREMENTS.md`.
- **Full room**: Wilson, Foreman, Chase, Cameron, Dr House, Health Inspector.
  83 claims audited, 4 struck, 5 unbound → `PROGNOSIS.md`.
- **Lean verified core**, `lake build CEQ` exit 0, four oleans, zero `sorry`.
  Lean 4.7.0 + mathlib.
- **R3 DELETED** — `tests/cameron/test_r3_perturbation.py` — attention 1565.111 vs
  Perron 396.868; baseline 3.94x sharper where R3 required flatter.
- **R4 DELETED** — `tests/cameron/test_r4_compression.py` — retained-filler control
  scores -0.11224 against crushed -0.11446; gap +0.00222 vs 0.25 pre-registered.
- **R6 DELETED** — `tests/cameron/test_r6_agreement.py` — averaging baseline wins by
  0.0793 AUROC; "averaged away" was a false premise.

## Iteration 1 — W2, the non-normal operator · DELETED, premise refuted

RED: `tests/w2/test_w2_nonnormal.py::test_non_normal_operator_beats_attention_on_perturbation_ratio[cpu|cuda]`
Suite: 14 passed, 2 failed (the falsifier). Both devices.

**Verdict: DELETE.** Causal non-normal ratio **788.077** against attention's
**1565.111** — factor 0.504, needed 3.0. Better than the deleted Perron arm's 396.868
by 1.99x, still half the baseline.

**The premise was false, and the instrument check is what proved it.** Henrici departure
from normality, median over 8 draws:

| operator | departure | ratio |
|---|---|---|
| row-stochastic gamma P | 1.40679 | 1618.510 |
| perron s_j P_ij | 1.40682 | 396.868 |
| causal nilpotent | 1.40972 | 788.077 |

Three operators within **0.21%** of each other on departure-from-normality, spanning
**4.08x** on the ratio. Non-normality is not the variable. The `monodromy` lead
(Henon lambda_1 = +0.42084 amplifying while det = 0.3 contracts) is true of that map and
does not transfer. Bound by
`test_non_normality_does_not_explain_the_perturbation_ratio`.

**Three results kept from a deleted item:**

1. `test_causal_nilpotent_is_better_conditioned_than_the_resolvent_arms` —
   rho(A) = **0.000000** exactly, so cond(I-A) = **48.92** against 771.12 and 747.95.
   A **15x** conditioning improvement, free, because nilpotency dodges the 1/(1-rho)
   blowup. No Perron certificate, no power iteration, so the measured reducible-A
   failure mode (min(w) = 6.6e-14, weighted norm undefined) cannot arise at all.
2. `test_resolvent_terminates_and_is_exact` — (I-A)^-1 = I + A + ... + A^(N-1)
   TERMINATES. Residual < 1e-12. This is `CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent`
   with its hypothesis A^N = 0 actually satisfied, so the sum IS the inverse rather than
   a truncation of bounded error.
3. `test_attention_buys_its_ratio_by_going_globally_dead` — attention reaches 1565.111 at
   content R2 **0.5577** and effective size **1.0087** of 32; the causal operator holds
   R2 **0.9951** at effective size **21.5281**. R3's third clause is "Global deadness
   fails" and the baseline fails it. This does NOT rescue W2 — its falsifier compares
   ratios and the causal operator loses 0.504x — but any restatement has to decide
   whether ratio-at-matched-content was the quantity meant.

Module written: `ceq/nonnormal.py`, standalone, imports nothing from the user's repos.

## Iteration 1 — W3, eviction as its own requirement · ALIVE

RED first: `ImportError: cannot import name 'eviction'`. Then
`tests/w3/test_w3_eviction.py` — **10 passed, cpu and cuda**.

**Verdict: ALIVE.** The first requirement to survive its own falsifier.

`test_eviction_forgets_where_gating_cannot` — perturb a token the gate crushes, keep-set
held fixed from the unperturbed context so the arms differ only in HOW the token is
removed:

| regime | max change in settled state (median of 24) |
|---|---|
| post-softmax gating | 2.154868e-05 |
| eviction | **0.000000e+00 — exact zeros, 24 of 24 draws** |

`test_gating_leaves_the_crushed_token_in_the_denominator` — the mechanism, isolated from
the settled state. Row-sum deficit `\|rowsum/rho - 1\|`, row 0 excluded:

| regime | deficit | reading |
|---|---|---|
| gating | 2.307863e-03 | leaked mass, never reallocated |
| eviction | 1.110223e-16 | machine epsilon; survivors renormalized |

A post-softmax gate cannot touch the softmax denominator. The crushed token keeps its
share of every surviving row's normalizer, so it shadows the state permanently and
perturbing it still moves every row. Eviction removes it before the softmax and the
survivors renormalize over what remains.

`test_salience_eviction_retains_the_token_that_matters` and
`test_compressed_state_still_spans_the_causal_content`, KEEP = 16 of 96:

| keep-rule | causal-token retention | subspace R2 of x[causal] |
|---|---|---|
| salience | **24/24** | **1.0000** |
| random | 5/24 | 0.4619 |
| recency | 3/24 | 0.4446 |

Both controls sit at the chance level for a 16-dim subspace of a 32-dim space (0.5).

**Honest limits on this result.** The salience R2 of 1.0000 is near-tautological: if the
causal token is kept, its content is a row of the state by construction. The load-bearing
number is the retention rate, 24/24 against 5/24 and 3/24. And the corpus is synthetic
with a planted control channel, so the gate has an easy job — this establishes the
MECHANISM (gating leaks, eviction renormalizes) and not that a learned gate finds causal
tokens in real text.

**Instrument correction made mid-iteration.** The first version of the content test
reconstructed `x[keep]` from `settle(x[keep])` — an invertible linear map of the thing
being reconstructed — and returned R2 = 1.0 for salience, random and recency alike. It
measured nothing. Replaced with a row-space projection of `x[causal]` onto the compressed
state, which is the question the requirement actually asks.

Module written: `ceq/eviction.py`. Standalone; takes the quotient-and-free-face idea from
`Epsilon-Hollow`'s `foliation` as inspiration and imports none of it.

## Iteration 2 — W3b, nilpotent resolvent proved in Lean · DONE

RED first: `FileNotFoundError: lean/CEQ/Nilpotent.lean`. Then
`tests/w3b/` — **6 passed**. `lake build CEQ` exit 0, zero `sorry`.

**`CEQ.Nilpotent.pow_card_eq_zero`** — a strictly lower-triangular
`A : Matrix (Fin n) (Fin n) R` satisfies `A ^ n = 0`.

Proof is path counting, not a matrix-theory citation. `A i j` is nonzero only
when `j < i`, so a nonzero entry of `A ^ k` needs a strictly increasing chain of
`k` indices and therefore `j + k <= i`. At `k = n` that demands an index of at
least `n`, which `Fin n` does not have. Induction on `k` with the invariant
`i < j + k -> (A ^ k) i j = 0`; the successor step splits each summand on whether
the last hop is on-or-above the diagonal (`A l j = 0`) or strictly below
(`j < l`, so the remaining `k` hops cannot reach `i` either).

**`CEQ.Nilpotent.occupancy_is_exact_inverse`** discharges the hypothesis of the
existing `CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` for the operator that
actually ships. Two consequences the numerical version does not get:

- No contraction certificate is consulted at all. `rho(A) = 0` holds
  structurally, so the reducible-`A` failure mode Chase measured — power
  iteration returning `w` with exact zeros at min 6.6e-14, leaving the weighted
  sup norm undefined — cannot arise, because no `w` appears in the statement.
- No truncation error. `weighted_contraction` bounds a tail; here the tail is
  identically zero.

**`CEQ.Nilpotent.one_not_nilpotent`** is the negative control: the identity is
lower triangular in the NON-strict sense and is nilpotent at no power. It exists
so the hypothesis cannot later be relaxed from `.tril(-1)` to `.tril(0)` by
someone reading only the theorem name.
`test_the_nilpotency_theorem_is_stated_over_the_shipped_operator` additionally
greps `ceq/nonnormal.py` for `.tril(-1)`, so the proof and the code cannot drift
apart silently.

**Two instrument faults found and fixed in my own tests this iteration**, both
recorded because a broken checker is worse than none:

1. The `sorry` detector split on `--` only and fired on CEQ.lean's own header
   sentence "No `sorry` anywhere" — it failed precisely when the code was
   correct. Replaced with a `/- -/`-aware stripper, and
   `test_the_sorry_detector_actually_detects_a_sorry` now calibrates it against
   a real `sorry`, a block comment, and a line comment.
2. `test_lean_library_builds_clean` reads `lake`'s own exit status directly. A
   backgrounded wrapper returned 0 earlier in this project while `lake` itself
   returned 1 with three real errors, and that false green was believed for one
   message.

Build fault worth noting: the first version failed with `CommRing ℝ` unsolvable
because `Mathlib.Data.Real.Basic` was not imported and `ℝ` was auto-bound as a
fresh universe variable. The two core theorems had compiled; only the
ℝ-instantiated ones failed.

Suite after this iteration: `tests/w2 tests/w3 tests/w3b` — **30 passed, 2
failed**. The 2 are the W2 falsifier, deliberately still RED as the record of
that deletion.

## Iteration 3 — W6.2a, signed causal attention, CPU + CUDA forward · DONE

RED first: `ImportError: cannot import name 'attention' from 'ceq'`, then a real bug in my
own causality test (`(q,k,v).index(tensor)` compares tensors with `==`). Final:
`tests/w6/` — **26 passed, cpu and cuda**.

**Foreman's R2 report landed mid-task and changed the design before it was written.**
The operator I was about to ship was `rho * softmax(...)` — non-negative, therefore tier 2,
therefore provably unable to represent a negation. It was replaced with a signed operator
before the first line of it existed.

### The tier-3 result

| operator | min influence Jacobian `d(out_i)/d(v_j)` |
|---|---|
| non-negative control (softmax, same construction) | **+0.000000e+00 — exactly zero** |
| signed operator (ships) | **−9.000000e-01** |

Foreman's general statement: the Kleene star of a non-negative matrix has a non-negative
influence Jacobian *in any ordered semiring*, measured as exactly 0.000e+00 over 40
max-plus instances and 160 APPNP kernels. `not` is unreachable by softmax attention, by
APPNP, and by the max-plus star alike. Dropping non-negativity is the only escape, and
this is it, measured on the operator that ships. Bound by
`test_signed_operator_reaches_negative_influence` with
`test_the_nonnegative_control_is_stuck_at_exactly_zero` as its calibration.

### Dropping non-negativity costs nothing, and iteration 2 is why

A signed operator has no Perron vector, so a Perron certificate cannot certify it —
Foreman flagged that as an unaddressed gap. It does not apply here.
`CEQ.Nilpotent.pow_card_eq_zero` is stated over `[CommRing R]` with the single hypothesis
`forall i j, i <= j -> A i j = 0`. **No non-negativity appears in it. Nilpotency is
sign-blind.** Measured on the signed operator: `rho(A) = 0.000e+00`, `A^S` exactly zero,
max row L1 exactly 0.900000. Bound by `test_nilpotency_does_not_care_about_sign`.

The Lean written in iteration 2 for a different reason is what licensed the iteration-3
design change. That was not planned.

### R1's deletion does not touch this design

Howard policy iteration beats value iteration 7.9x on CPU and 21.4x on CUDA because value
iteration is a fixed-point SEARCH. A nilpotent path sum is a direct method: finite, exact,
nothing to converge toward, no policy to improve. `path_sum` says so in its docstring.

### Truncation is a cost knob with a stated bound

| K | actual tail | stated bound `rho^(K+1)/(1-rho)` |
|---:|---:|---:|
| 2 | 8.137e-01 | 7.290e+00 |
| 4 | 1.156e-01 | 5.905e+00 |
| 8 | 3.749e-04 | 3.874e+00 |
| 16 | 8.715e-13 | 1.668e+00 |

Valid at every K and **loose by roughly 50x** at K=4. Reported rather than tightened.

### Honest cost, C6

CPU float32, B=2 H=3 S=24 Dh=16: dense causal SDPA **0.0874 ms**, ceq at hops=4
**0.8069 ms** — **9.2x**, against a naive K=4 prediction of 4x. The excess is the explicit
`[S,S]` operator materialization against SDPA's fused kernel. This is the number that makes
a real kernel worth writing rather than optional.

### Both interfaces registered

`AttentionInterface.register` **and** `AttentionMaskInterface.register`. Registering only
the first makes transformers pass `attention_mask=None` and silently drop causal, padding,
packing and sliding-window constraints. `test_both_interfaces_are_registered` checks both
mappings.

Causality is verified by **output parity** against the unedited run — bitwise zero leakage
for edits to q, k and v alike — never by inspecting a mask, because a mask was already
measured reporting the correct sparse pattern while the kernel computed fully dense.

Module: `ceq/attention.py`. Completion condition 1 is met on CPU and CUDA.

## Iteration 4 — W4, intervention generalization · SPLIT VERDICT

`tests/w4/` — **19 passed, 2 failed**. The 2 are the absolute falsifier, correctly RED.

### The ordering is real and reproduces

OOD NRMSE on the held-out composition, median of 5 seeds, matched parameters (2,625 each),
matched budget, identical data:

| arm | tier | train NRMSE | OOD NRMSE |
|---|---|---|---|
| softmax attention | 1 | 0.2861 | **5.8198** |
| APPNP (control) | 2 | 0.2047 | **4.2107** |
| ceq signed path sum | 3 | 0.1274 | **2.6151** |

Signed beats APPNP in **4/5 seeds**, median ratio **0.6770** against a 0.80 threshold,
paired difference **-1.4427 +/- 0.9531**, and with lower variance (+/-0.4252 vs +/-1.0069).
The predicted tier ladder appears in the predicted direction. Bound by
`test_the_tier_ordering_is_the_predicted_one`.

### The generalization claim is FALSE, for every arm

**NRMSE 1.0 is predict-the-mean. All three arms are above it.** Every arm is worse than a
constant predictor on the held-out composition. Tier 3 fails 1.6x less badly than tier 2.
It does not generalize.

R2 as specified says "must generalize under intervention where a similarity baseline
fails". That bar is not met. Bound RED by
`test_signed_arm_generalizes_rather_than_merely_degrading_less`.

**The weak falsifier was mine to get wrong.** I first wrote a RELATIVE separation test
(`signed < 0.8 * appnp`), which passes, when the requirement asks an ABSOLUTE question.
The ordering result must never be reported without the absolute test beside it.

### A fatal corpus bug, found and fixed mid-iteration

The first corpus split OOD by literal VALUE -- train 1..3, test 7..9 -- so the test tokens
indexed embedding rows that training never updated. Every arm landed near predict-the-mean
(attention 0.954, appnp 0.779, signed 0.777) and the run measured embedding coverage, not
generalization. Replaced with a split by COMPOSITION: every `(op, flag)` cell except
`('-', 1)` is trained, that cell alone is tested, literals 1..9 on both sides.
`test_the_held_out_cell_is_a_composition_of_seen_tokens` now blocks that class of error.

A second instrument was also wrong: `test_the_negation_gate_produces_negative_coupling`
regressed the outcome on the flag linearly. The gate flips a SIGN, so its marginal
coefficient averages to zero over a corpus symmetric in `op` -- the test failed while the
corpus was behaving correctly. Replaced with a direct check that flipping the flag negates
the executed outcome.

### Corpus

`ceq/corpus.py`. Oracle is CPython, no answer key anywhere in the file.
`test_exec_agrees_with_a_subprocess_interpreter` proves the `exec` speed shortcut did not
change the oracle.

## Iteration 5 — W7, Nash-equilibrium stance · DELETED

`tests/w7/` — 14 passed, 4 failed. The 4 are the kill condition and the
beats-signed test, both correctly RED.

**Verdict: DELETE.** The equilibrium machinery is correct and does what it says. It does
not buy composition.

### The machinery works

All fourteen structural tests pass on cpu and cuda: the QRE reaches a genuine fixed point
(best-response residual 0.000e+00), the stance stays in [0,1] with no projection, the
operator is signed and strictly causal with row L1 <= rho, nilpotency survives the stance
multiplication (`rho(A) < 1e-12`, `A^S` exactly zero), and the update is non-affine at
every probe radius rather than piecewise — Foreman's surviving R1 criterion, met.

### The result does not

| arm | OOD NRMSE, median of 5 seeds |
|---|---|
| signed path sum (W6) | **2.6151** |
| nash @ tau=4.0 | 3.6183 |

Nash beats signed in **1/5** seeds. Paired difference **+0.7520 +/- 1.1582** — worse.
Below the absolute 1.0 bar in **0/5**.

### A single-seed result nearly shipped

A tau sweep on seed 0 alone showed tau=4.0 at OOD **2.3444**, beating signed's 3.7686,
with `safe_tau` at 5.2730 and lower taus worse. That looked like a real optimum and a real
tension between uniqueness and expressiveness. Across five seeds it evaporates: 1/5.

This is the failure mode already on record from the physics-prior round, where a
first single-seed run passed and the null control caught it. Here the control was seeds.
Reported because the near-miss is the lesson, not the verdict.

### Two by-products worth keeping

1. **`safe_tau` is sufficient, not necessary.** The contraction threshold is
   `||M||_2 / 4 = 8.048`. QRE converged to residual **0.000e+00** at tau = 10.060, 4.000
   AND 1.000 — the last at Lipschitz constant 8.048, eight times above the bound. The
   guarantee is conservative and the iteration is better behaved than it certifies.
2. **The uniqueness/expressiveness tension is real but not decisive.** At `safe_tau` the
   stance range is [-0.2744, +0.9292]; at tau=1.0 it is [-0.9999, +1.0000]. Buying
   guaranteed uniqueness costs most of the signed range. That trade exists; it is not what
   decided this verdict, because no tau reached the bar on more than one seed.

`ceq/nash.py` is kept, not deleted from disk: its tests document a negative result and its
`affine_fit_ratio` probe is reusable. Nothing imports it in the live path.

## Iteration 6 — W8, the survivors against a real checkpoint · DONE

`tests/w8/` — **20 passed**, cpu and cuda. Model: `Qwen/Qwen2.5-0.5B`, fp32, GQA.

### The adoptability bar, met exactly

A signed operator swapped into softmax-trained weights destroys the model, so any
perplexity measured that way is measuring the swap. The adoptable form is a residual gate,
`out = stock_attention(q,k,v) + sum_h (alpha A)^h v`, which is BITWISE stock at alpha = 0.

| alpha | perplexity | vs stock |
|---|---|---|
| stock | 9.2150 | -- |
| 0.00 | **9.2149** | **-0.00%** |
| 0.01 | 9.1779 | -0.40% |
| 0.05 | **9.0781** | **-1.49%** |
| 0.15 | 9.1789 | -0.39% |
| 0.30 | 10.7969 | +17.17% |

Greedy generation at alpha = 0 is character-identical to stock. The gate is free when
closed, on both the teacher-forced and the decode path.

**The -1.49% at alpha = 0.05 is ONE 99-token sample and is not a result.** It is recorded
because it is what was measured, not because it supports anything. A perplexity delta on
untuned weights says the gate is wired correctly; C3 stands and perplexity is not a win
condition.

### Two bugs found, both invisible to the tests that were passing

**1. Output layout.** `AttentionInterface` requires `[B, S, H, D]`; `sdpa_attention_forward`
ends with `.transpose(1, 2).contiguous()`. Returning `[B, H, S, D]` folds heads and
sequence into each other, raises nothing, and produced **perplexity 89400.180 against
1.667** for the untouched checkpoint. The parity test passed throughout, because it
compared the gated path against this file's own `stock_attention` -- both wrong the same
way. Internally consistent, externally garbage. Only parity against the real model caught
it. Same shape as the `acc / l_i` broadcast bug and the `from_kv_blocks` trap. Now pinned
by `test_output_layout_is_batch_seq_heads_dim`.

**2. Decode-time causality, which perplexity cannot see.** `is_causal=True` is correct only
when `q_len == kv_len`. During incremental decoding the query is one token against a cached
`kv_len = N`, and torch aligns the causal mask to the top-left of a 1xN grid, so the single
query attends to position 0 alone. Teacher-forced perplexity never decodes: it matched
stock to **0.0000%** at alpha = 0 while greedy generation produced
`' The following the 1: 1: 1: 1:'` against stock's `' The capital of France is Paris.'`.
**A generation check is not optional.** Now bound by
`test_gate_closed_generates_identically_to_stock`.

### A structural limitation, refused rather than papered over

The multi-hop path sum **cannot** be computed incrementally from a standard KV cache. During
decode the operator row is `[1, N]`, so `A^2` is undefined. Falling back to `hops=1` would
make the module behave differently in prefill and decode without saying so, so alpha > 0
during decode raises `NotImplementedError` instead, and the message names the fix:

    (A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j

Caching the h-th hop vector per position makes decode exact at K times the value-cache
memory and K attention rows per token. **Not built.** Bound by
`test_multihop_decode_is_refused_not_silently_downgraded`.

### One fragile test of mine, replaced

`test_the_cost_of_the_operator_on_untuned_weights_is_measured` asserted that perplexity
rises monotonically with alpha. It passed on the fixture text and is false in general --
the curve above is non-monotonic. Asserting a property that merely happens to hold on the
fixture is how a fragile test survives to mislead later. Replaced with what is actually
true: finite everywhere, free when closed, degraded at large alpha.

## Iteration 7 — W9, the scoped island hop cache · ALIVE

`tests/w9/` — **20 passed**, cpu and cuda. `ceq/hopcache.py`.

Four structures, each supplying exactly one thing a flat KV cache does not:

| source | contributes |
|---|---|
| the filtration `(A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j` | decode at all -- K rows/token, K slots |
| NeMo-Relay `scope_stack.rs` | LIFETIME: LIFO scopes, pop drops exactly that scope, root refuses removal |
| `mujoco#3396` island discovery | GRANULARITY: disjoint-set H0 of the coupling graph, min-index representative, no `n x n` scratch |
| foliation | ADMISSIBILITY: only a free face may be evicted |
| W3 (measured here) | EXACTNESS, with the scope corrected below |

### The wall is down

alpha > 0 can now decode. `hop_h[i]` depends only on `hop_{h-1}[j]` for `j < i`, so caching
one vector per hop per position makes incremental decoding possible at K attention rows per
token and K slots of memory. `test_cache_holds_k_slots_not_k_squared` pins that the tower
is a filtration and not a product.

### Bitwise was the wrong claim for parity, and the right one for eviction

**Prefill vs decode is NOT bitwise and asserting it would have been dishonest.** Prefill
contracts the sequence in one matmul; decode contracts one row at a time. Same value,
different summation order. Measured, float64, S=32: max absolute **1.665335e-16**, mean
1.320947e-17, relative **1.898657e-16** -- one ULP. The operator ROW agrees to
**2.775558e-17**, so the entire gap is in the hop accumulation and none of it is in the
operator. The test asserts a relative bound and quotes the measurement.

**Eviction IS bitwise, but only in a narrower window than W3's result.** Two tests now
separate what was one wrong assertion:

- `test_eviction_is_bitwise_when_nothing_dependent_has_run` -- evict before anything reads
  the position and the result is identical to never inserting it. Bitwise.
- `test_eviction_is_not_retroactive` -- evict after dependents have run and it is NOT.
  Their hop vectors already contracted the contribution into stored state, and removing the
  key cannot unwind it.

**This is the one place the cache is weaker than the operator.** W3 measured eviction
bitwise at 0.000e+00 over 24/24 draws, but that was the OPERATOR, where the whole
computation is redone from the retained key set. In a cache the contribution is already
stored. The exact remedies are to recompute the affected hop vectors -- O(N), which is what
the cache exists to avoid -- or to evict before anything reads the position. Stated, not
hidden.

My first version of that test asserted the strong form, failed, and was correct to fail.

## Iteration 9 — W11, the README's claims must resolve · DONE

`tests/w11/` — **6 passed**. Completion condition 3.

A document full of test names that do not resolve is worse than one with none: it reads as
evidence and is not. This checks `README.md` against pytest's **actual collected node ids**
rather than a hand-maintained list, because a test sitting in a file pytest cannot import is
not a test and a grep would report it as one.

Six checks, each aimed at a way the document could drift away from the repo:

| check | what it catches |
|---|---|
| `test_the_collector_actually_finds_known_tests` | the instrument itself — an empty collection would pass every other check vacuously |
| `test_readme_exists_and_leads_with_limits` | house style, checked POSITIONALLY: the Limits heading must appear before the results section |
| `test_every_test_name_in_the_readme_resolves` | a claim with no evidence wearing the costume of one |
| `test_the_reproduce_commands_are_real` | a reproduction command that does not run |
| `test_deleted_requirements_are_stated_as_deleted` | the document quietly keeping six requirements while the repo has two |
| `test_the_failed_intervention_result_is_in_the_limits` | the most important NEGATIVE result being buried below the fold |

The last one is the point. It pins "all three arms are worse than a constant predictor",
with the numbers 2.6151 and 4.2107, inside the Limits section specifically -- so a later
edit cannot quietly move the failure somewhere a reader will not reach.

Written because this project has already produced three checkers that were internally
consistent and externally wrong: the parity test that compared the gated path against this
repo's own `stock_attention` (perplexity 89400.180 against 1.667); the `sorry` detector that
fired on CEQ.lean's own sentence "No `sorry` anywhere"; and the eviction test that asserted
bitwise invisibility in a window where it is provably false. Each one passed while being
wrong. This file calibrates itself before trusting itself.

No device parametrization, deliberately: test collection has no device axis, and a fake one
would run the identical subprocess twice and report two passes for one fact.

## Iteration 10 — W10 probe, the pure operator vs softmax from scratch · NOT AT PARITY

Reduced-budget probe, CPU, byte-level TinyStories, d=128 / 2 layers / 4 heads / seq 64 /
bs 16 / 250 steps / lr 1e-3, identical for both arms:

| arm | params | train | val | wall-clock |
|---|---|---|---|---|
| softmax | 3,319,296 | 1.8528 | **1.8838** | 69.9 s |
| signed | 3,319,296 | 2.2740 | **2.2965** | 107.6 s |

`signed / softmax = 1.2191` against a parity bar of 1.05. **NOT AT PARITY.**

**The gap is worse than the ratio suggests.** The signed arm runs `hops=3`, three matmuls
against softmax's one, and spent **54% more wall-clock** to lose by 22%. It had more compute,
not less. There is no iso-FLOP correction that helps it here.

Both arms beat uniform `ln(256) = 5.5452` comfortably, so the deadness guard holds and this
is a comparison between two models that both learned -- not two that both failed.

### The verdict this forces

W10's kill condition was stated in advance: *if the pure operator cannot train to parity
with softmax at matched budget, the module is a correction term and not a replacement, and
every claim about it must be restated in those terms.*

**It is a correction term.** `README.md` now says so in the Limits section, above every
result, and `tests/w11/` pins it there.

That verdict does not retract anything measured. The signed operator still reaches an
influence-Jacobian minimum of **-9.000e-01** where every non-negative operator -- softmax,
APPNP, the max-plus star -- is stuck at exactly **0.000000e+00**, and that capability is
real and is not available any other way. What is retracted is the framing: it is a
capability ADDED to attention, not a superior attention.

### Honest scope of this number

A reduced-budget probe, not the formal run. The full-budget comparison (600 steps, d=256,
4 layers) is `tests/w10/ -m slow` and was still running when this was recorded. A 1.2191x
gap against a 1.05 bar is large rather than marginal, so the direction is not in doubt, but
the exact figure will be replaced by the formal one.

Neither arm was hyperparameter-tuned. That is symmetric and matches C3's matched-tuning
requirement, but it does mean the signed arm has not been given its best shot -- whether the
`rho * w / sum|w|` normalization needs a different learning rate is exactly what Chase is
measuring in `tests/chase/test_signed_operator_trainability.py`.

## Iteration 11 — W12, the model card · DONE. Completion conditions 3 and 4 met.

`tests/w11/` — **11 passed**. `MODEL_CARD.md` written, house style, limits first.

Five checks specific to the card, each aimed at a way it could oversell:

| check | what it prevents |
|---|---|
| `test_model_card_exists_and_leads_with_limits` | results appearing above limits, checked positionally |
| `test_every_test_name_in_the_model_card_resolves` | a cited test that does not exist |
| `test_the_model_card_says_it_is_not_a_replacement` | W10's verdict being omitted or buried below the capability claim |
| `test_the_model_card_declares_no_weights` | an HF card being read as a checkpoint when it is a module |
| `test_the_single_sample_perplexity_is_flagged_as_such` | the -1.49% at alpha=0.05 being printed without its one-sample caveat |

Completion condition status: **1 met, 2 met, 3 met, 4 met.** The promise is still not output,
because meeting the four conditions is not the same as the artifact being what was aimed at.
It is an honest correction term. It was aimed at a replacement.

## Iteration 12 — W10 FORMAL RESULT · the gap widens with training

`tests/w10/ -m slow`, 850.9 s, cpu and cuda:

| budget | softmax val | signed val | ratio |
|---|---|---|---|
| 250 steps, bs 16 | 1.8838 | 2.2965 | 1.2191x |
| **600 steps, bs 32** | **1.5780** | **2.1103** | **1.337x** |

**A correction I owe on my own probe.** I set `lm.D_MODEL = 128` and `lm.N_LAYERS = 2`
after import, intending a smaller probe architecture. Those are default-argument bindings
evaluated at `def` time, so the mutation never took: both rows above ran the SAME
architecture, d=256 / 4 layers / seq 128. Only batch size and step count differ.

That accident makes the comparison cleaner than intended, and the signal is bad:
**more training widens the gap.** Softmax exploits its selectivity as budget grows; the flat
L1-normalized operator saturates. A gap that closes with budget would be an optimization
problem. One that widens is an expressiveness problem.

Both README.md and MODEL_CARD.md now carry both rows, and `tests/w11/` asserts the presence
of "1.337", "1.2191", "1.5780", "2.1103" AND the phrase "WIDENS with training" inside the
Limits section -- so a later edit cannot quietly drop the direction while keeping the number.

## Iteration 10 — FOREMAN: is the operator new? · TWO OF THREE CLAIMS COLLAPSE

Three RED files, `tests/foreman/`, 20 RED assertions and 12 GREEN, cpu and cuda.

### Q1 — the operator is a triangular solve · REDUCED

RED: `tests/foreman/test_operator_is_a_triangular_solve.py`

| claim as written | measured | what it is |
|---|---|---|
| not forward substitution | **1.776e-15** | `(I-A)z = v`, Golub & Van Loan Alg. 3.1.1 |
| not a Neumann iteration | **8.882e-16** | K+1 steps of `z <- Az + v`, already in `_lib.neumann_iterate` |
| not a linear-attention RNN | **1.479e-14** rel | Katharopoulos arXiv 2006.16236, state accumulating OUTPUT not value |
| is a Volterra series | **0.000e+00** | ruled out: `f(tx) = t f(x)`, degree ONE, not 2K+1 |

The `d x d` recurrent form runs in O(S d^2). The single thing forcing quadratic cost is the
`sum_j |q_i . k_j|` normalizer, because `|.|` does not accumulate into a linear state --
`test_the_absolute_value_in_the_normalizer_is_the_only_non_recurrent_part`.

**Nilpotency is not new either.** DeltaNet Eq. (10), arXiv 2406.06484, inverts
`I + tril(diag(beta) K K^T, -1)` -- strictly lower triangular and SIGNED (min entry
-10.875878) -- and `fla/ops/utils/solve_tril.py` documents it as "the inverse of the matrix
I + A. A should be strictly lower triangular". Neither DeltaNet paper mentions nilpotency;
sglang's KDA prefill kernel does, verbatim: `(I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8)
[L strictly lower triangular, L^16=0]`. The module's own `path_sum` reproduces that inverse
to **2.183e-10** -- `test_the_finite_path_sum_inverts_the_deltanet_matrix_too`.

### Q2 — the obstruction is real and is NOT the one predicted

RED: `tests/foreman/test_l1_normalizer_obstruction.py`

**The conditioning hypothesis is refuted.** Total gradient norm at identical initialization
on identical TinyStories bytes: signed **1.8342e+00** against softmax **1.8820e+00**, ratio
**0.9767**, matched within 2.3%. The `1/||w||_1` rank-one-correction blow-up is not there.

**The real obstruction is an exact rank deficiency.** `A = rho w / sum|w|` is homogeneous of
degree ZERO, so `A(t q, k) = A(q, k)`:

| logit scale | 0.25 | 1.0 | 4.0 | 16.0 |
|---|---|---|---|---|
| signed, peak row weight | 0.100336 | 0.100336 | 0.100336 | 0.100336 |
| softmax, same tensors | 0.0504 | ... | ... | 1.0000 |

Spread **0.000e+00** across a 64x sweep. `grad_q . q = 1.088e-14` against softmax's
**10.561**. Every query vector permanently loses one of its d degrees of freedom, row L1
mass is pinned at rho, and there is no temperature channel at all. **This is the mechanism
behind W10's widening gap**: softmax exploits selectivity as budget grows because it can
scale its logits; this operator cannot, so it saturates. A gap that widens with budget is an
expressiveness problem, and here is the expressiveness that is missing.

Two exact zeros come with it. `A[1,0] = +/- rho` identically -- one predecessor, so the
normalizer cancels the logit -- with gradient **0.0e+00** and a **2 rho = 1.8** jump where
its logit crosses zero. And `w/||w||_1` is DISCONTINUOUS at `w = 0`, not merely non-smooth.

### Q3 — signedness alone is published prior art · NOVELTY COLLAPSES

RED: `tests/foreman/test_signedness_is_not_new.py`

Reimplemented from defining equations, same instrument, same draw:

| operator | min influence Jacobian |
|---|---|
| softmax control | +0.000000e+00 |
| **SimA, arXiv 2206.08898 (2022)** | **-3.929583e-01** |
| **DiffAttn, arXiv 2410.05258, lambda=0.8** | negative |
| ceq path sum | -9.000000e-01 |

SimA L1-normalizes Q and K in place of softmax and states the property in its own text:
"the attention values can become negative, meaning that a token can affect another one
negatively. This is in contrast to regular transformers where the attention is always
non-negative." That is the tier-3 claim, published in 2022.

**And the headline number is the frozen entry.** -9.000e-01 is -rho exactly. Row 1 is the
only place an influence entry can equal -rho, because every other row splits rho across two
or more predecessors and a path sum of entries summing to rho is strictly inside rho. Over
24 draws: **14 of 14** entries equal to -rho sat at (1,0); most negative anywhere else
**-8.845515e-01**. The advertised property is read off the one entry with zero gradient.

**What is left.** The general semiring statement is correct and the surviving unoccupied
combination is narrow: a **row-L1-normalized signed QK operator used with a multi-hop sum**.
Every multi-hop propagation found (APPNP 1810.05997, GDC 1911.05485, MAGNA 2009.14332)
requires a non-negative matrix; every signed attention found (SimA, Differential Transformer,
Cog Attention 2411.07176, FAGCN 2101.00797, SignGT 2310.11025) is single-hop. DeltaNet is
the near miss -- signed, strictly triangular, exactly inverted -- but as a chunkwise
parallelization device, not as the attention.

README.md and MODEL_CARD.md restated. STATE.md W6 moved ALIVE -> RESTATED.

## Iteration 13 — the parity campaign, first pass · 1.337 -> 1.0968

Every row below is 3 seeds on CUDA, 600 steps, d=256/4 layers/seq 128, identical
parameters (3,319,296), identical initialization, identical data. Softmax median **1.6227**.

| operator | median val | ratio | spread |
|---|---|---|---|
| signed, L1-normalized (the shipped one) | 2.1295 | **1.3124** | — |
| sgate, lam=1, hops=3 | 1.9545 | 1.2099 | 1.1959-1.2159 |
| sgate, lam=0.25, hops=3 | 1.8062 | 1.1181 | 1.0997-1.1369 |
| **sgate, lam=0.25, hops=2** | **1.7881** | **1.0968** | 1.0950-1.1149 |

**74% of the gap closed. Parity bar is 1.05, so this is NOT parity.**

### Three separable obstructions, each measured

**1. No temperature channel.** `A = rho*w/sum|w|` is homogeneous of degree ZERO:
`A(t*q, k) = A(q, k)`. Peak row weight was **0.100336 at logit scales 0.25, 1.0, 4.0 and
16.0 alike -- spread exactly 0.000e+00 over a 64x sweep** -- against softmax 0.0504 ->
1.0000. `grad_q . q = 1.088e-14` against softmax's 10.561. Every query permanently loses a
degree of freedom. That is the mechanism behind the widening W10 gap, and it refutes the
flatness framing in favour of something sharper. Foreman, `tests/foreman/test_l1_normalizer_obstruction.py`.

Fix: both halves of `sgate` are softmaxes, which are not degree-zero homogeneous.
**1.3124 -> 1.2099.**

**2. DC annihilation.** At `lam = 1` both halves sum to 1, so rows sum to exactly zero and
`A` annihilates the constant vector -- measured row sum **0.000000e+00**, `|A @ 1| =
1.192e-07`. The path sum can then add only deviations, never signal level. The lam sweep is
perfectly monotone in row sum:

| lam | row sum | ratio |
|---|---|---|
| 0.00 | 0.9000 | 1.1099 |
| 0.25 | 0.5400 | 1.1181 |
| 0.50 | 0.3000 | 1.1493 |
| 0.75 | 0.1286 | 1.1818 |
| 1.00 | 0.0000 | 1.2099 |

**Read honestly: lam=0 is the best point and it is NON-NEGATIVE.** Tier 3 is gone there. So
signedness costs monotonically on this task, 1.1099 -> 1.2099. lam=0.25 is the compromise
that keeps the property at 1.1181.

**3. hops.** Weakly non-monotone: 2 beats both 1 and 3 at lam=0.25. **1.1181 -> 1.0968.**

### What remains

Even at `lam = 0` -- fully non-negative -- the ratio is **1.1099**. So the multi-hop path
sum costs ~11% independent of sign. That residual is the next target and it is not a
signedness problem.

## Iteration 14 — H7, the identity double-count · DELETED, refuted

RED first, then measured. 3 seeds, CUDA, 600 steps, lam=0.25, hops=2, softmax median 1.6227:

| kind | val median | ratio | spread | s/run |
|---|---|---|---|---|
| sgate (identity kept) | 1.7881 | **1.0968** | 1.0950-1.1149 | 32.7 |
| sgate_nores (identity dropped) | 2.0821 | **1.2887** | 1.2568-1.2889 | 21.2 |

**Verdict: DELETE.** Removing the identity term made it substantially WORSE, 1.0968 ->
1.2887.

**My reasoning was wrong and here is where.** I argued that `Block.forward` is
`x = x + attn(n1(x))` while the path sum returns `v + Av + ...`, so the signed arms carried a
duplicated residual the softmax control did not. They do not. The block residual carries the
**pre-projection hidden state**; the `v` inside the path sum passes through `self.proj`.
Different paths, different transforms. `v` is a genuine value-attention term -- it is what
attention with an identity coupling would produce -- not a copy of the residual.

The structural tests were all correct and all passed: the k=0 term IS exactly `v`, dropping
it does change the output, the operator stays signed, strictly causal and bounded, and no
parameters move. The hypothesis about what that term MEANT was the wrong part.

By-product worth keeping: `sgate_nores` is 35% faster (21.2s vs 32.7s per run) because it
runs one fewer accumulation. It is also 18% worse. Not a trade worth taking.

## Iteration 14 — Chase's chain report · corrections to shipped documents

**The memory figures in README.md and MODEL_CARD.md described INFERENCE.** Training
forward+backward against SDPA at identical parameters, fp32: 1.44x / 1.74x / **2.65x /
4.45x / 8.06x** at seq 128/256/512/1024/2048. The 1.31-1.94x on record is the forward-only
kernel in the 128-512 band only. Both documents corrected.

**A 0.5B signed model fits no Colab GPU at seq 2048 batch 1.** 36.31 GiB against a 33.53 GiB
A100-40GB budget; L4-24GB has 20.25 GiB. The softmax control fits both. Cause: SDPA never
forms `[S,S]`; this operator materializes it and autograd retains a measured **3.9 tensors of
`[B,H,S,S]` per layer**, so activations are O(S^2) against O(S). Gradient checkpointing is
therefore mandatory, not optional -- measured **0.379x** (1287.5 -> 487.7 MiB), loss identical
to 1e-4, and it buys batch 15 on an A100 and 7 on an L4.

**`register_for_auto_class` silently breaks weight tying.** `PreTrainedModel.is_remote_code()`
is literally `cls._auto_class is not None`, so the single call that makes `save_pretrained`
copy the `.py` files routes the model onto a branch that strips the tied target out of
`missing_keys`; `tie_weights` then treats it as already present and `lm_head.weight` is never
loaded off the meta device. Nothing raises. **The two things a custom architecture on the Hub
needs are mutually exclusive in transformers 5.3.0.** Rollback shipped:
`tie_word_embeddings=False`, costing 40.96M params on 515.57M = +7.9%.

**My NaN prediction was FALSE.** 0/800 non-finite gradient norms on real data, both arms,
peak signed grad norm 1.10x the control. The real hazard is a conditioning spread hidden
inside one global clip: row L1 spans 1.10e-03 to 1.67e+01 at initialization (1.5e+04x), and
over 800 steps the minimum reached 9.35e-07 while the median rose 16.55 -> 53.07. Worst and
typical rows move in opposite directions. Rollback `eps=1e-3` bounds `|dA/dw|` by rho/eps and
is off by default, since `eps=0.0` is bitwise the shipped operator.

**bf16 autocast does not halve activation memory** -- 0.748/0.772/0.797 at seq 256/512/1024,
2.89-3.36 effective bytes per element rather than 2. Correcting his own optimistic assumption
moved the A100 verdict from "batch 1" to "batch 0".

**One thing got better: a free Colab T4 can run this.** The Triton kernel is forward-only so
it cannot appear in a training step; the training path is pure torch, so sm_80 and the
Linux-only wheel do not gate Colab training at all.

## Iteration 15 — H8, rho is the self/other ratio · ALIVE, and the campaign reaches 1.0258

`tests/w15/` — 4 structural passed. All numbers 3 seeds, CUDA, 600 steps, matched params.
Softmax median **1.6227**, 15.6 s/run.

### rho had never been swept

With `lam = 0` the operator is `rho * softmax(w)` with the DIAGONAL EXCLUDED, and the
output is `v + Av + ...`. The identity supplies the token's own value at coefficient exactly
**1**; every other token is capped at total weight `rho`. **The self/other balance is pinned
at `1 : rho`**, and softmax learns that balance freely because its diagonal is just another
logit. `rho = 0.9` was chosen in the first hour and inherited by every measurement since.

**rho > 1 is legal, and that is not a technicality.** `pow_card_eq_zero` needs only
`forall i j, i <= j -> A i j = 0` and bounds nothing about magnitude, so `A^n = 0` at any
rho. Verified at rho = 0.9, 1.5 and 3.0: `A^n` exactly zero, spectral radius < 1e-12. What
rho > 1 costs is the truncation bound `rho^(K+1)/(1-rho)` -- a statement about truncating an
INFINITE series. This series is finite. The bound was never load-bearing here.

| rho | ratio | spread |
|---|---|---|
| 0.50 | 1.1800 | 1.1726-1.1991 |
| 0.90 | 1.0968 | 1.0950-1.1149 |
| **1.50** | **1.0625** | 1.0412-1.0659 |
| 2.50 | 1.0811 | 1.0785-1.0865 |

### A correction I owe on my own last report

I wrote "signedness costs monotonically" from the lam sweep at rho = 0.9. **It does not hold
at rho = 1.5.** Measured there:

| lam | ratio | min influence | min A |
|---|---|---|---|
| 0.00 (non-negative) | 1.0676 | 0.0000e+00 | 0.0000 |
| 0.05 | 1.0586 | -3.3067e-02 | -0.0408 |
| **0.10** | **1.0529** | -7.2663e-02 | -0.0781 |
| 0.15 | 1.0552 | -1.1385e-01 | -0.1139 |

**The non-negative control is WORSE than the signed optimum.** Signedness helps at rho=1.5
and hurt at rho=0.9; the two knobs interact and I reported one slice as if it were general.

### The campaign, and where it stopped

| configuration | ratio |
|---|---|
| signed, L1-normalized (shipped) | 1.3124 |
| sgate lam=1 hops=3 rho=0.9 | 1.2099 |
| sgate lam=0.25 hops=3 rho=0.9 | 1.1181 |
| sgate lam=0.25 hops=2 rho=0.9 | 1.0968 |
| sgate lam=0.10 hops=2 rho=1.5 | 1.0529 |
| **sgate lam=0.10 hops=1 rho=1.5** | **1.0258**, spread 1.0164-1.0281, 20.1 s/run |

### Why the promise is NOT being output

`1.0258 <= 1.05`, 3 seeds, spread and wall-clock reported, matched parameters, operator
signed (min A = -0.0781). The literal condition is met.

**It is met only at hops = 1, and at hops = 1 this is not the module.** The path sum
degenerates to `v + Av` -- a single hop. Foreman established that
`(softmax(A1) - lambda * softmax(A2))V`, single-hop with lambda per head, is Differential
Transformer, arXiv:2410.05258. So parity at hops=1 is a **replication of a published
method**, not a result about the multi-hop signed path sum this campaign exists to defend.
At hops=2 the same settings give 1.0529 and at hops=3 they give 1.1634 -- multi-hop is where
the module's identity lives and it is not at parity there.

Ending a 50-iteration campaign on that would be technically true and substantively hollow.
The loop continues, and the target is restated: **parity at hops >= 2.**

### Instrument NOT trusted

I wrote a sign-flip probe in this iteration and it returned 0.0000 at hops 1, 2 AND 3 --
including configurations where Cameron's calibrated `bench.sign_flip_rate` measures 0.0469
and 0.1875 for the `signed` operator. Mine is uncalibrated, single-entry, written in one
pass. **No conclusion is drawn from it.** Cameron's instrument does not yet support `sgate`;
extending a calibrated tool is the correct next step and was not done hastily at the end of
an iteration.

## Iteration 16 — PARITY REACHED at hops=2, with the property intact

### The instrument was calibrated before the new arm was trusted

`ceq/bench.sign_flip_rate` extended to `sgate`. Calibration reproduced every published
number **exactly**: signed d1/h3 **0.0469**, signed d2 **0.1875**, softmax **0.0000** at both
depths, softmax_gelu d2 **0.0547**. Only then was the new arm read.

| arm | depth | hops | content-conditional sign rate |
|---|---|---|---|
| softmax (control) | 1, 2 | any | **0.0000 exactly** |
| signed (shipped L1) | 1 | 3 | 0.0469 |
| **sgate** | 1 | 1 | 0.0234 |
| **sgate** | 1 | **2** | **0.1641** |
| sgate | 1 | 3 | 0.1484 |

Two things this settled. `sgate` carries the property at **3.5x** the shipped operator's
rate at hops=2. And it carries it at hops=1 too (0.0234), where the L1 operator is exactly
zero -- so last iteration's worry that hops=1 is property-less was wrong for this operator.

### The fair move was the learning rate, not a fifth operator knob

Neither arm's `lr` had ever been tuned. C3 requires a matched tuning budget, so the same
grid was swept for both and each arm's own best was taken:

| lr | softmax | sgate h2 | ratio |
|---|---|---|---|
| 1e-4 | 2.1734 | 2.2135 | 1.0185 |
| 3e-4 | 1.6227 | 1.7009 | 1.0529 |
| **1e-3** | **1.2957** | **1.3349** | **1.0334** |
| 3e-3 | 1.3221 | 1.5149 | 1.1458 |

`lr = 1e-3` is the best point for **both** arms, so this is best-to-best and not a
cherry-pick.

### Confirmation, 5 seeds

`rho=1.5, lam=0.10, hops=2, lr=1e-3`, 600 steps, CUDA, byte-level TinyStories:

| seed | softmax | sgate | ratio |
|---|---|---|---|
| 0 | 1.2918 | 1.3349 | 1.0334 |
| 1 | 1.3024 | 1.3319 | 1.0227 |
| 2 | 1.2957 | 1.3492 | 1.0413 |
| 3 | 1.2855 | 1.3111 | 1.0199 |
| 4 | 1.2774 | 1.3240 | 1.0365 |

**median 1.0334, spread 1.0199-1.0413, under the 1.05 bar on 5/5 seeds.**
Parameters **3,319,296 = 3,319,296**, exactly equal. Operator signed, `min A = -0.1080`.
Wall-clock **21.3 s/run against 16.1 s/run = 1.32x**.

### The campaign, end to end

| configuration | ratio |
|---|---|
| signed, L1-normalized (as shipped, lr 3e-4) | 1.3124 |
| sgate lam=1 hops=3 rho=0.9 | 1.2099 |
| sgate lam=0.25 hops=3 rho=0.9 | 1.1181 |
| sgate lam=0.25 hops=2 rho=0.9 | 1.0968 |
| sgate lam=0.10 hops=2 rho=1.5 | 1.0529 |
| **sgate lam=0.10 hops=2 rho=1.5 lr=1e-3** | **1.0334** |

Four separable obstructions, each measured: no temperature channel (degree-zero
homogeneity, spread exactly 0.000e+00 over a 64x logit sweep); DC annihilation (row sums
exactly 0 at lam=1); the self/other ratio pinned at `1:rho` with rho never swept; and an
untuned learning rate shared by both arms.

### Honest limits on this result

**The signed arm had four knobs tuned; softmax had one.** `rho`, `lam`, `hops` and `lr`
against `lr` alone -- because softmax has no operator knobs to tune. That is the fair
comparison in the sense that each arm got its own best, and it is an asymmetry in tuning
surface that a reader should know about.

One dataset (TinyStories bytes), one size (3.3M, d=256, 4 layers, seq 128), one budget (600
steps). **This is parity at 3.3M parameters, not a scaling claim.** The W10 finding that the
gap WIDENS with budget was measured on the L1 operator and has not been re-measured on
`sgate`; whether this parity holds at longer training is untested and is the first thing to
check before any 0.5B run.

## Round 3 — FOREMAN: the scaling verdict, the leap attempt, and ParaFormer

The standard applied here is the user's, not the loop's: **parity at 3.3M is FALSE unless it
survives at 300M.** Everything below is bound to a test in `tests/foreman/test_sgate_scaling_defect.py`,
every test parametrizes over cpu and cuda.

### The instrument was recalibrated before anything was read

`n_draws` for every published `sign_flip_rate` number is **128**, recovered exactly from the
values themselves: 0.0469 = 6/128, 0.1641 = 21/128, 0.0234 = 3/128. Reproduced on cpu and
cuda. `test_the_sign_flip_probe_still_reports_its_published_numbers` GREEN.

### FINDING 1 — the distinguishing property is a short-context property, `s^-1.4`

0.1641 was measured at the probe's default context of **s = 8**. Swept, relative positions
held fixed (`i = s-1, j = s/4, c = s/2`), 1024 draws:

| s | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| sgate | 0.17480 | 0.08887 | 0.02637 | 0.01172 | **0.00391** |
| softmax | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

log-log slope **−1.389**, R² **0.9938**. At the 1024–2048 context a 300M run uses, that is
2e-4 to 8e-5 — under the resolution of 10,000 draws.

Controls, all run: discard floor disabled reproduces it; `lam = 0` (negative half deleted)
reads exactly 0.0 at every s, like softmax; the shipped L1 `signed` operator decays
identically 0.0957 → 0.0039; no `rho` in (0.5, 1.5, 4.0) and no `lam` in (0.05 … 2.0) holds
half its own s=8 rate to s=128 -- the test's grid, 256 draws, s=8 -> s=128: lam 0.05
0.140625->0.007813, 0.25 0.160156->0.000000, 1.00 0.128906->0.000000, 2.00
0.144531->0.000000; rho 0.5 0.140625->0.003906, 4.0 0.167969->0.003906; d_head 32
0.183594->0.000000, 64 0.132813->0.003906, 128 0.128906->0.000000. Depth 2 and 3 do not
restore it.

**Head width does not touch it, and head width is the axis that grows from 3.3M to 300M.**
At d_head 16 / 32 / 64 / 128 the rate at s=128 is 0.00000 / 0.00000 / 0.00000 / 0.00000
(192 draws) while s=8 holds at 0.16667 / 0.17708 / 0.13021 / 0.13542. Wider heads make it
slightly worse at s=32: 0.03646 -> 0.00521 from d=16 to d=64.

**NOT attention fading.** Negative-entry fraction of `A` flat at 0.4390 → 0.4673 over
s = 8 → 1024; `min A` pinned at −0.1364 = −ρλ/(1+λ) throughout; both branches stay at
0.05–0.12 of the uniform entropy ceiling out to s = 1024. The hypothesis this round opened
with — that `softmax(-w)` degenerates to a prefix mean — is **refuted by its own test**,
which passed.

**The cause: the property and the decay are one fact.** A third token `c` reaches the pair
`(i,j)` only along a path through `c`, and the first term of `J = Σ_k A^k` containing one is
`k = 2`. Measured by the test at 512 draws, hops = 1: sgate **0.017578**, pairwise-sign
**0.000000**, against **0.123047** and **0.083984** at hops = 2. The property is carried entirely by the two-hop term, which sums over ~s
intermediates of which `c` is one. Share = 1/s, by construction.

**Softmax does the same wherever it has a sign to lose.** On `wrt="x"`, the sign-unconstrained
input path, softmax reads 0.025391 / 0.009766 / 0.000000 at s = 8/32/128 on cpu and
0.023438 / 0.009766 / 0.000000 on cuda, 512 draws. So the decay is
dilution of one token's leverage, common to attention; what is specific to the signed
operator is only that it has the property on the *value* path at all.

RED: `test_claim_content_conditional_sign_survives_a_longer_context`,
`test_claim_some_knob_in_the_operator_family_arrests_the_context_decay`,
`test_claim_the_property_does_not_live_only_in_the_multi_hop_term`,
`test_claim_the_context_decay_is_specific_to_the_signed_value_path`.

### FINDING 2 — the leap was attempted and it FAILED its own falsifier

If the sign is decided by comparing two *globally normalized* quantities it must dilute. So:
decide the sign from an **unnormalized pairwise** quantity, take only the magnitude from a
normalized one.

    A = rho * sgn(w) * softmax(|w|)

Row L1 exactly rho (2.4e-07 float deviation), strictly causal (0.0), 49.17% negative entries,
`CEQ.Nilpotent.pow_card_eq_zero` applies unchanged. This is Cog Attention's matrix
(arXiv:2411.07176 Eq. 3, `SignExp(p)/Σ|SignExp|` = `sgn(p)·softmax(|p|)`) — published, and
**single-hop**, so the path sum over it is exactly the cell this project called unoccupied.

Measured, 512 draws: **0.0840 / 0.0332 / 0.0137 / 0.0039 / 0.0020** at s = 8/16/32/64/128.
It decays the same way. `test_claim_a_pairwise_sign_holds_the_property_where_a_normalized_one_loses_it`
RED. **The decay belongs to multi-hop signed propagation, not to the normalizer.**
`ceq/bench.py::_causal_signmag_operator`.

### FINDING 3 — ParaFormer is REAL, and the residual novelty is now MEASURED rather than asserted

arXiv:2512.14619, *ParaFormer: A Generalized PageRank Graph Transformer for Graph
Representation Learning*, Yuan, Song, Kuruoglu, Zhao, Liu, Zhao, Cheng, Rong, submitted
**16 Dec 2025**. Verified by direct fetch, not by nurse report. Eq. 9 verbatim:

    Z = Σ_{k=0..K} γ_k Â^k V = Σ_k γ_k (Softmax(QK^T))^k V

"Here, {γ_k ∈ ℝ | k=0,1,2,⋯,K} is a set of learnable weights of GPA"; Theorem 1 uses
`γ_k = (−a)^k/2, a ∈ (0,1/n), k > 1`. `Â = Softmax(QK^T/√d)` — **non-negative base**.

So README's "every published multi-hop propagation requires a non-negative matrix" was FALSE
and is corrected. Signed *matrices* raised to powers are also occupied — BernNet 2106.10994
(`L^k`), ChebNetII 2202.03580 (`T_k(L̂)`), PCNet 2403.03676 (`(−L̃)^n`) — but all of a fixed
structural Laplacian, never a content-dependent QK matrix.

The residual claim narrows to **signed base matrix, content-dependent** — and it survives on
the instrument. ParaFormer's Eq. 9 reimplemented and probed: **0 flips in 2048 draws**,
exact Clopper-Pearson 95% ceiling **0.00146**, against sgate's **0.1230** on the same draws.
At least 84×. Signed coefficients decide the sign by where a non-negative hop profile falls
against the fixed hyperplane `Σ γ_k x_k = 0` — second order in a third token; a signed matrix
carries a sign per entry — first order.
`test_claim_a_signed_hop_coefficient_cannot_do_what_a_signed_matrix_does` GREEN (RED first as
`ValueError: paraformer`). `ceq/bench.py::sign_flip_rate` gains a `paraformer` arm whose
gammas are drawn from a stream used by no other arm, so every published number is unchanged.

### FINDING 4 — a 3% val-loss gap is not parity at 1-3B, and it is measured

arXiv:2605.20798 (20 May 2026), verified by direct abstract fetch. 20 post-2021 modifications
at 1.2B and 3B, iso-data, iso-compute, iso-recipe, multi-seed noise floor. Verbatim: "two
significant failures converge to within **2-3% of baseline validation loss** yet drop 6-16
CLIMB-points"; "1.2B improver rank is a weak predictor of 3B improver rank", Spearman
ρ = −0.27, ranks moving 7→1, 10→3, 1→6. This campaign's headline gap is **3.34%**, measured
three orders of magnitude below where that decoupling was characterised. Tay 2207.10551
abstract, verbatim: "the best performing model can fluctuate at different scales."

### OPEN — stated, not bound to a test

- **The decay is measured on RANDOM projections, not trained ones.** `sign_flip_rate` draws
  `wq`, `wk`, `wo` from `N(0,1)`. The argument that it transfers is combinatorial and not
  measured: a third token's share of the two-hop sum is `1/s` regardless of what the
  projections learned, and the fading confound is excluded independently (rows stay peaked,
  `max p+` 0.9179 → 0.7903 over s = 8 → 1024). Binding it needs the probe run through a
  trained model's own attention, which is plumbing this round did not build.
- **`signmag` has never been trained.** `ceq/bench.py::_causal_signmag_operator` exists, is
  well-formed, and failed the sign-decay falsifier. Its validation-loss ratio against softmax
  is unknown. It may be a better operator that happens not to fix the property.
- **The exponent is −1.389, not −1.** The 1/s argument predicts −1. Where the extra 0.39
  comes from is not established. Candidates not separated: the two-hop sum concentrating
  (CLT) so `|J_ij|` has less density near zero as s grows; the `d = 16` head width held fixed
  while s grows.
- **No capability probe exists at any context length.** Every number in this campaign is a
  validation-loss ratio or a Jacobian-sign rate. `arXiv:2605.20798` is the measurement that
  those two families of number stop predicting each other at 1-3B, and this repository has
  nothing on the other side of that gap.
- **The 0.1641 headline has a wide interval.** It is 21/128 draws; the binomial 95% interval
  is roughly [0.10, 0.24]. Re-runs at n_draws 64 / 256 give 0.1719 / 0.1445 for the same
  configuration. The decay is far outside that noise; the headline itself is not a precise
  number and has never been reported with one.

## Iteration 19 — Foreman round 3: the property does not survive context. NO LEAP.

The most important report of the run, and the catch was one I should have made.

**`0.1641` was measured at the probe's default context `s = 8`. Nobody swept `s`.**

| s | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| sgate | 0.17480 | 0.08887 | 0.02637 | 0.01172 | **0.00391** |
| softmax | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Log-log slope **-1.389**, R^2 **0.9938**. At 1024-2048 context: 2e-4 to 8e-5, under the
resolution of 10,000 draws.

**Mechanism, and it is the elegant part.** A third token `c` reaches the pair (i,j) only via
a path *through* `c`; the first term of `J = sum A^k` containing one is `k=2`. Measured
hops=1: 0.0273 / hops=2: 0.1445. The two-hop term sums over ~`s` intermediates of which `c`
is one, so its share is **1/s by construction**. **Multi-hop creates the property and
multi-hop dilutes it.** They are the same fact.

Three candidate mechanisms I would have guessed were all killed by their own falsifiers:
negative-branch entropy (stays 0.05-0.12 of ceiling to s=1024), fixed row L1 (`min A` pinned
at exactly -0.1364 = -rho*lam/(1+lam) from s=8 to s=1024), rank (negative-entry fraction flat
0.4390 -> 0.4673). The operator is exactly as signed at s=1024 as at s=8.

**Nothing arrests it**: lam 0.05-2.00, rho 0.5/1.5/4.0, depth 2 and 3, head width
16/32/64/128 -- all 0.00000 at s=128.

**Honest control**: on the input path softmax decays too. This is ordinary dilution of one
token's leverage. What is specific to the signed operator is only that it has the property
on the VALUE path where softmax is exactly zero -- an infinite ratio over a vanishing
absolute.

### The leap: attempted, falsified

`A = rho * sgn(w) * softmax(|w|)` -- sign from an unnormalized pairwise quantity, magnitude
from a normalized one. Correct diagnosis, correct move, and it **decays identically**:
0.0840 / 0.0332 / 0.0137 / 0.0039 / 0.0020. It is also Cog Attention's matrix
(arXiv:2411.07176 Eq. 3). **The normalizer was never the problem.** Left in place as
`ceq/bench.py::_causal_signmag_operator`; never trained, so its loss ratio is unknown.

### ParaFormer is REAL and the README was wrong

arXiv:2512.14619, ParaFormer, 16 Dec 2025. Eq. 9 verbatim: `Z = sum_k gamma_k A_hat^k V`
with learnable `gamma_k`. **"Every published multi-hop propagation requires a non-negative
matrix" was FALSE.** BernNet, ChebNetII and PCNet also raise signed matrices to powers --
but always a fixed structural Laplacian, never a content-dependent QK matrix.

**The residual claim survives the instrument.** ParaFormer reimplemented on the calibrated
probe: **0 flips in 2048 draws**, exact Clopper-Pearson 95% ceiling **0.00146**, against
sgate's 0.1230 on the same draws -- **>= 84x**. Signed *coefficients* decide the sign by
where a non-negative hop profile falls against the fixed hyperplane `sum gamma_k x_k = 0`,
which a third token moves only to SECOND order; a signed *matrix* carries a sign per entry,
FIRST order. That distinction is real, novel, and now measured against real prior art.

### The win condition itself is inside the noise band

arXiv:2605.20798, verified by direct fetch: *"two significant failures converge to within
2-3% of baseline validation loss yet drop 6-16 CLIMB-points"*, Spearman rho = **-0.27**
between 1.2B and 3B improver rank. The campaign's 3.34% target sits inside the band where
val loss stops predicting capability.

### And the headline never had an interval

0.1641 is 21/128. Binomial 95% interval approx **[0.10, 0.24]**. Re-runs: 0.1719, 0.1445.

## Round 4 - FOREMAN: the 84x was an instrument bug, and the last claim is deleted

The build round. Every number below is bound to `tests/foreman/test_paraformer_ratio_across_context.py`,
which parametrizes over cpu and cuda; cuda reproduces cpu exactly at s = 8 and s = 128.

### FINDING 1 - the ParaFormer arm never ran ParaFormer

`ceq/bench.py::sign_flip_rate` built the `gam` vector for the `paraformer` arm and then never
referenced it. `op_kind == "paraformer"` was not in the path-sum branch, so the arm executed
`a = _softmax_operator(qq, kk)` followed by `h = a @ h` -- plain single-hop softmax. Round 3's
"0 flips in 2048 draws, exact Clopper-Pearson 95% ceiling 0.00146, at least 84x" was a
**softmax** number.

RED: `test_the_paraformer_arm_is_not_the_softmax_arm` -- at `hops=0` the two arms consume the
identical random stream, and ParaFormer truncated to its `k = 0` term is `Z = gamma_0 V`,
blind to `x`. It read **0.02734375 == 0.02734375**, bitwise the softmax rate. GREEN after the
arm was routed through `sum_k gam[k] A^k h`. Two further guards were needed: `allow_unused`
for a leaf absent from the graph, and a `requires_grad` check for an output with no graph at
all -- both are exact zeros, not errors.

`test_claim_a_signed_hop_coefficient_cannot_do_what_a_signed_matrix_does` in
`test_sgate_scaling_defect.py` flipped GREEN -> RED as a direct consequence: the corrected arm
reads **0.023926**, not 0.0, against sgate's 0.145020 at 2048 draws each. Left RED.

### FINDING 2 - the corrected separation is 5.6x, not 84x

hops = 2, 2048 draws, seed 0, `i = s-1, j = s/4, c = s/2`, cpu. Exact Clopper-Pearson 95%.
Conservative ratio = `lo(sgate) / hi(paraformer)`.

| s | sgate | ParaFormer | point ratio | conservative ratio |
|---|---|---|---|---|
| 8 | 0.166016 [0.150143, 0.182850] | 0.029785 [0.022858, 0.038097] | 5.6x | 3.94x |
| 16 | 0.078613 [0.067323, 0.091127] | 0.009277 [0.005595, 0.014450] | 8.5x | **4.66x** |
| 32 | 0.027344 [0.020720, 0.035363] | 0.002930 [0.001076, 0.006366] | 9.3x | 3.25x |
| 64 | 0.008789 [0.005217, 0.013855] | 0.000488 [0.000012, 0.002717] | 18x | 1.92x |
| 128 | 0.003418 [0.001375, 0.007030] | 0.000000 [0, 0.001800] | inf | 0.76x |
| 256 | 0.000000 [0, 0.001800] | 0.000488 [0.000012, 0.002717] | 0 | 0.00x |
| 512 | 0.000000 [0, 0.001800] | 0.000000 [0, 0.001800] | -- | 0.00x |

softmax is 0 of 2048 at every s. `test_claim_the_separation_from_signed_hop_coefficients_survives_context`
RED at s = 128.

**At 2048 draws the ratio collapses; at 16,384 it does not.** The instrument runs out, not
the effect:

| s | sgate | ParaFormer | conservative ratio |
|---|---|---|---|
| 128 | 64/16384 = 0.003906 [0.003010, 0.004985] | 2/16384 = 0.000122 [0.000015, 0.000441] | **6.83x** |
| 256 | 18/16384 = 0.001099 [0.000651, 0.001736] | 1/16384 = 0.000061 [0.000002, 0.000340] | **1.92x** |
| 512 | 5/16384 = 0.000305 [0.000099, 0.000712] | 0/16384 = 0.000000 [0, 0.000225] | 0.44x |

The point ratio GROWS with context -- 5.6x, 8.5x, 9.3x, 18x, 32x at s = 8..128 -- which is the
direction round 3's first-vs-second-order argument predicts. The conservative bound does not
follow it: 3.94, 4.66, 3.25, 1.92 at 2048 draws over s = 8..64, then 6.83, 1.92, 0.44 at
16,384 draws over s = 128..512. It never exceeds 7x, it is not monotone, and by s = 512 it is
below 1 even at 16,384 draws -- both numerators are falling toward zero faster than any
affordable number of draws can track. Log-log slopes
over s = 8..512: sgate **-1.437** (R2 0.9959), ParaFormer **-1.261** (R2 0.8614). A ratio
that grows over two absolute rates that both vanish is not a capability.

### FINDING 3 - the residual cell is occupied by a shipping kernel, and it WINS

DeltaNet, arXiv:2406.06484 `eq:inverse`, verified by direct LaTeX-source fetch:

    T = (I + tril(diag(beta) K K^T, -1))^-1 diag(beta)

Strictly lower triangular, built from K and beta so content-dependent, entries
`-beta_i (k_i . k_j)` with no non-negativity so signed per entry, inverted exactly. Strictly
triangular is nilpotent, so that inverse **is** a finite path sum; arXiv:2606.06034 writes it
as `(I-A)^-1 = sum_n A^n`. Shipping in `fla/ops/utils/solve_tril.py` ("Compute the inverse of
the matrix I + A. A should be strictly lower triangular") and sglang's
`chunk_kda_fwd_kernel_inter_solve_fused`, wired into `RadixLinearAttention`. Neither the
paper nor either codebase uses the words "nilpotent" or "Neumann" -- 0 grep hits.

Same probe, same draws, hops = 2:

| s | sgate | **deltanet** | signmag | paraformer |
|---|---|---|---|---|
| 8 | 0.166016 | 0.082031 | 0.105469 | 0.029785 |
| 16 | 0.078613 | **0.097656** | 0.035156 | 0.009277 |
| 32 | 0.027344 [.020720, **.035363**] | **0.059082** [**.049264**, .070185] | 0.013184 | 0.002930 |
| 64 | 0.008789 | **0.021484** | 0.004395 | 0.000488 |
| 128 | 0.003418 | **0.010254** | 0.000977 | 0.000000 |
| 256 | 0.000000 | **0.003906** | 0.000000 | 0.000488 |
| 512 | 0.000000 | **0.000977** | 0.000000 | 0.000000 |
| slope | -1.437 | **-1.107** | -1.651 | -1.261 |

Three non-overlapping 95% wins for the shipping kernel: at s = 32, 2048 draws, DeltaNet
[0.049264, 0.070185] against sgate [0.020720, 0.035363], 1.39x; at s = 128, 16,384 draws,
0.011108 [0.009560, 0.012834] against 0.003906 [0.003010, 0.004985], 1.92x; at s = 256,
16,384 draws, 0.003784 [0.002903, 0.004849] against 0.001099 [0.000651, 0.001736], 1.67x.
It leads at every s >= 16, decays the slowest, and is the only arm still nonzero at s = 512.
`test_claim_the_signed_content_dependent_path_sum_is_not_already_deltanet` RED.

### FINDING 4 - a non-vanishing regime EXISTS, and it corrects round 3's mechanism

Round 3: the decay is the third token's share of the two-hop sum, "1/s by construction". That
sweep moved `j = s/4` and `c = s/2`, so the intermediate count and every softmax row
normalizer grew with s together. Pin the offsets at `j = i-4, c = i-2` and the path count
between j and i is a constant 3 at every s.

hops = 2, seed 0, w = 8. sgate at 2048 draws, others at 1024:

| s | sgate w=8 | sgate global | deltanet w=8 | deltanet global | paraformer w=8 | paraformer global | softmax |
|---|---|---|---|---|---|---|---|
| 32 | 0.124023 | 0.021484 | 0.074219 | 0.074219 | 0.019531 | 0.002930 | 0.000000 |
| 128 | 0.133301 | 0.001953 | 0.076172 | 0.076172 | 0.031250 | 0.000000 | 0.000000 |
| 512 | 0.108398 | 0.000000 | 0.065430 | 0.065430 | 0.019531 | 0.000000 | 0.000000 |
| 2048 | 0.120605 | -- | -- | -- | -- | -- | -- |

**Flat across a 64x growth in context** -- every windowed sgate interval overlaps every other
one out to s = 2048. `test_a_bounded_receptive_field_holds_the_separation_that_global_attention_dilutes`
GREEN on cpu and cuda.

**But the cause is the row normalizer, not the path count.** At a fixed path count of 3,
sgate on a global row still falls 0.018555 -> 0.000000 over s = 32 -> 512 (1024 draws), while
DeltaNet holds 0.074219 -> 0.065430 and is **bit-for-bit identical windowed and unwindowed**
-- it has no row denominator, so there is nothing for the context to dilute. That is the same
fact as its -1.107 slope and its win at every s >= 16.
`test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer` RED.

The regime is real and belongs to nobody: windowing rescues every normalized arm (ParaFormer
0.002930 -> 0.019531 at s = 32), leaves softmax at exactly 0, and does nothing for the arm
that already had no normalizer.

### The residual claim, in one sentence a reviewer can check

> On a strictly-causal random-projection influence-Jacobian probe at two hops, this module's
> signed content-dependent path sum separates from softmax absolutely (0.166016 [0.150143,
> 0.182850] against 0 of 2048, ceiling 0.001800, at context 8) and from ParaFormer's signed
> hop coefficients by a conservative 4.66x at context 16 and 6.83x at context 128 with 16,384
> draws -- but DeltaNet's shipping WY matrix is a signed content-dependent path sum too and
> beats this module at every context length from 16 up, non-overlapping at 32, 128 and 256,
> so the construction is not new; and every signed arm decays at about s^-1.1 to s^-1.7 on a
> global row, so none of it is measurable at the context a 300M model runs at.

**Nothing in this repository is novel attention.** What is left is an operator with a
Lean-verified nilpotent resolvent and a kernel that runs.

### Prior art added this round, verified by direct fetch

- **SignGT**, arXiv:2310.11025, 17 Oct 2023 -- `sgn(Q K^T) exp(|Q K^T|) / sum exp(|Q K^T|)`,
  signed and content-dependent, **earlier than Cog Attention** (2411.07176, 11 Nov 2024) and
  the same matrix as this repo's `signmag`. Its powers are of the fixed adjacency, `W = A^k`,
  not of the signed attention.
- **Signed Dual Attention**, arXiv:2606.04833, v1 3 Jun 2026 -- Eq. 1/2 verbatim:
  `A+ = softmax(QK^T/sqrt(dk))`, `A- = softmax(-QK^T/sqrt(dk))`, `SDA(Q,K,V) = (A+ - A-)V`.
  That is this campaign's `sgate` matrix at `lam = 1`. No lambda or rho appears anywhere in
  that paper (0 grep hits over its LaTeX); it states "equal weighting" and lists adaptive
  weighting as future work. Single-hop. **The sgate MATRIX is not new either.**
- **Differential Transformer is NOT the same object** and this repository has been sloppy
  about it. arXiv:2410.05258 Eq. 1 splits `[Q1;Q2] = XW^Q`, `[K1;K2] = XW^K` and differences
  `softmax(Q1K1^T/sqrt d) - lambda softmax(Q2K2^T/sqrt d)` -- two DISTINCT logit matrices from
  separate projection halves, not one logit matrix and its negation. SDA is the exact match;
  DiffTransformer is a cousin.
- **InfSA**, arXiv:2603.00175 -- content-adaptive Neumann series `(I - gamma A)^-1 - I`, but
  `A = ReLU(QK^T)` normalized, non-negative.
- **arXiv:2606.06034** -- writes Gated DeltaNet's `(I-A)^-1 = sum_n A^n` explicitly; states
  "the entries of A are bounded in [0,1]", so negativity is unattested there.
- ParaFormer's gamma_k is confirmed a **scalar per hop**, `{gamma_k in R | k=0..K}`, not
  per-head, per-channel or content-dependent; `Ahat = Softmax(QK^T/sqrt(d))` is its only
  definition and no signed-base variant is defined or ablated; K = 10 in the experiments.

### OPEN - stated, not bound to a test

- **Every cross-arm comparison is UNPAIRED.** The `paraformer` arm draws its gammas and the
  `deltanet` arm draws its `beta` before `x0`/`v0`, so that no other arm's random stream --
  and no previously published number -- moves; round 3's calibration still reproduces exactly
  (signed 0.046875, sgate hops=2 0.1640625, hops=1 0.0234375, softmax 0.0 at 128 draws). The
  price is that those two arms do not see the same `x0`, `v0` as `sgate`. At 2048 and 16,384
  draws that is a variance argument, not a bias one, but no test enforces it.
- **The DeltaNet arm is a reimplementation, not the kernel.** `beta = sigmoid(N(0,1))`, keys
  L2-normalized, `diag(beta)` postfactor dropped because a per-row positive rescale cannot
  move a sign, and the path sum truncated at hops = 2 rather than run to nilpotency. Every
  one of those choices is defensible and none is the shipping code. Binding it needs the fla
  kernel in the loop.
- **Every arm is measured on RANDOM projections.** Unchanged from round 3, and the row
  normalizer finding makes it sharper: a trained model's row denominator is not a sum over
  `s` iid logits, and whether the `s`-dependence survives training is unmeasured.
- **The window result is measured at one width.** `w = 8` only. Whether the rate scales as
  `1/w` -- which the normalizer story predicts -- is untested, and it is the cheapest
  remaining experiment.
- **The conservative ratio is not monotone and nobody knows why.** 3.94, 4.66, 3.25, 1.92 at
  2048 draws, then 6.83 and 1.92 at 16,384. The point ratio grows cleanly; the bound does not.
  Separating "resolution" from "real non-monotonicity" needs draws this round did not spend.
- **No capability probe exists at any context length.** Unchanged from round 3.
- **`signmag` still has never been trained.** Unchanged from round 3, and it now has a 2023
  citation (SignGT) as well as the 2024 one.

---

## Round 12 — ΔFloor-by-eviction DELETED. The Lean theorem that was built to rescue it is what killed it.

### The instrument was wrong first, and the first verdict is void

`scale/dfloor_probe.py` kept `min(4, nblk)` blocks with `nblk = s//32`. At s=64 that is
2 of 2 blocks and at s=128 it is 4 of 4 — **every block kept**, so recall was 1.000 by
construction at two of its three points. Its own output printed the tell: `chance for
top-4 of s/32 blocks: s=64: 2.000`. A chance level above 1 means the arm cannot fail.
The reported slope −0.708 was fitted through two structural constants and one number.

Eighth instrument in this project that was internally consistent and externally wrong,
and the second — after the multizoom Δ=−0.424 — whose result was True at every point
tested by construction. **The v1 verdict is void in both directions**: it neither killed
nor rescued anything.

### The corrected instrument, calibrated

`scale/dfloor_probe2.py` holds the KEEP FRACTION at 0.25 instead of the block count, so
chance is 0.250 at every s. Calibration: the `random` arm reads 0.323 / 0.271 / 0.281 /
0.229 at s = 128 / 256 / 512 / 1024, every CI containing 0.250. The instrument reads
chance when it should.

| selector | s=128 | s=256 | s=512 | s=1024 | slope |
|---|---|---|---|---|---|
| **dfloor** | 0.292 [0.203,0.393] | **0.167** [0.098,0.256] | 0.156 [0.090,0.245] | **0.146** [0.082,0.233] | **−0.309** |
| cosine | 0.260 [0.176,0.360] | 0.354 [0.259,0.458] | 0.344 [0.250,0.448] | 0.302 [0.213,0.404] | +0.060 |
| random | 0.323 [0.231,0.426] | 0.271 [0.185,0.371] | 0.281 [0.194,0.382] | 0.229 [0.150,0.326] | −0.143 |

n = 96 draws per cell, Clopper-Pearson intervals, chance = 0.250 flat.

**Both pre-registered kill numbers fire.** recall at s=256 is 0.167 ≤ 0.48; slope −0.309
< −0.1. Neither was moved after seeing data. At s=1024 the dfloor CI is [0.082, 0.233],
**entirely below chance** — the score is not uninformative, it is anti-correlated.

And the resemblance control this project exists to beat is the only arm that holds up
with scale: `cosine` is flat-to-rising and above chance at every s.

**ΔFloor-by-eviction is DELETED under C2.**

### `CEQ.Refcount` — proved, and it is what transfers the kill

The proposed rescue was to replace the leave-one-out forward pass with an O(1) refcount
read, on the ground that they are the same number. `lean/CEQ/Refcount.lean` proves they
are, and builds against mathlib v4.7.0 with no `sorry`; all five theorems depend only on
`propext`, `Quot.sound`, `Classical.choice`.

- `floor_add_orbits` — `Σ_plaques (refcount − 1) + m = n`. Theorem 1 of `caustic` read
  off the fibre cardinalities of `foliation`.
- `evict_floor_add_refcount_pred` — post-eviction floor + (refcount − 1) = pre-eviction
  floor.
- `free_face_floor_unchanged` — refcount 1 ⇒ the floor is unmoved. Eviction safety is a
  theorem, not a measurement.
- `shared_plaque_floor_drops` — the converse, so the criterion is not vacuous.

**A first draft of the ΔFloor theorem was FALSE and `omega` refused it.** Stated as
arithmetic on `(n, m, k)` it fails at `n=3, m=3, k=3`: the left side is 0 and the right
truncates to 2. That triple is unreachable in a real cache, but nothing in the arithmetic
said so. Restating over the actual surviving prefix set (`survivors`) makes the missing
constraint a consequence of the structure instead of an assumption — `image_survivors` is
that step. The additive discipline is what caught it.

### Why the theorem kills the rescue rather than saving it

`refcount` is a property of **cross-sequence prefix sharing** and of nothing else.
Measured: in a single sequence at s=256 and s=1024, every block-aligned prefix occurs
exactly once, so every refcount is 1, every score `refcount − 1` is **0**, and the set of
distinct score values has size **1**. The selector is a coin flip by construction. Only
with four sequences sharing a 512-token prefix does anything exceed 1 (refcounts `[1, 4]`,
16 of 80 plaques shared — exactly the shared prefix).

So the refcount-priced schedule is **not a retrieval or ranking mechanism at all**. It is
a safety certificate for deduplication: "a block whose prefix is referenced once can be
dropped bitwise-exactly." That statement is true, now proved, and is what prefix caching
already does. It is not the object that was wanted.

### Wins that survive this round

- `CEQ.Refcount` is real, novel as a linking statement between two of the author's own
  results, and machine-checked. It is a *certificate*, not a *selector*, and must be
  described as one.
- The corrected probe is now a calibrated instrument with a flat chance baseline and can
  be pointed at any future selector.

### Losses recorded

- ΔFloor-by-eviction as a KV selector: DELETED, both kill numbers, calibrated instrument.
- The refcount-priced schedule as a *selector*: dead on arrival — score is constant within
  a sequence. Survives only as an eviction-safety certificate.
- `scale/dfloor_probe.py` v1: void, ceilinged by construction. Superseded by v2.

## Round 4 — CHASE: `ceq/hf/` is publishable, and four silent failures are gone

The package now implements the operator the number is about, every hazard raises or is
counted, and one CPU-only command checks the whole chain. Everything below is bound to a
test in `tests/chase/test_hub_package_hardening.py`; every test parametrizes over cpu and
cuda. **58 passed** there, and `tests/chase/test_ceq_hub_package.py` 22 passed / 2 xfailed,
`tests/w6` 26, `tests/w8` 20, `tests/w11` 11.

### The package did not implement the operator the number is about

`modeling_ceq.ceq_operator` was `rho*w/||w||_1` at rho=0.9 hops=3. The 1.0334 median is
`sgate` at rho=1.5 lam=0.10 hops=2. Relative difference **1.818e+00**, on record since round
3 in `test_scale_hazards.py` and never acted on. `sgate_operator` now ships and the default
config is the parity point, checked **bitwise** against `ceq/lm.py` where the 5-seed run ran.
`signed` stays selectable as the negative control at its own rho=0.9 hops=3.

`ceq/attention.py` contains no sgate implementation, so it cannot be the drift oracle for the
shipped default. The two operators have two different oracles and the docstring now says so.

### Four silent failures, each measured before it was fixed

| hazard | measured | now |
|---|---|---|
| additive mask of `-1e4` | operator `torch.equal` to the UNMASKED one; masked columns kept **0.48290979862213135** of weight | `ValueError` naming the convention; `STATS` counts masked key positions |
| chunked prefill, q_len=4 kv_len=16, no mask | query 0 moved **1.8396726846694946** when key 15 was perturbed | `NotImplementedError`; 1-token decode and masked chunks still pass |
| `tie_word_embeddings=True` | `lm_head.weight` on META, forward RUNS, logits[0,0,0] = **1.0053620544046395e+30**, max abs diff **1.0053620544046395e+30** against the model that was saved, nothing raised | `RuntimeError` naming the meta device |
| `truncation_bound(1.5, 2)` | returned **-6.75**, a negative error bound, at the shipped operating point | `ValueError`; there is no geometric bound at rho >= 1 |

A fifth was **created** by changing the default and caught before it shipped: a `config.json`
written before the `operator` key carries `"rho": 0.9, "hops": 3` and no operator, and
`from_dict` calls `cls(**config_dict)`, so it would have loaded as `sgate` at rho=0.9 hops=3
on weights trained with `signed`. Setting any operator knob without naming the operator now
raises. `ceq/hf/train.py::build` carried the same trap in its own signature (`hops=3,
rho=0.9`) and no longer defaults the operator knobs at all.

### A correction I owe on my own round-6 report: alpha=0 was not bitwise on GQA

"alpha = 0 is BITWISE stock" was asserted against `ceq.hybrid.stock_attention` — this
repository's own SDPA — not against the attention the model runs. Against the real `sdpa`
path on a randomly-initialized Llama: **0.000e+00 at 4 kv heads and 1.490e-07 max abs /
2.609e-07 relative at 2 kv heads.** Cause: `_repeat_kv` materializes the expanded keys while
transformers passes `enable_gqa=True` to torch, and the two reduce in different orders —
**4.768e-07** apart on bare tensors. **Qwen2.5-0.5B is a GQA model**, so the 9.2149-vs-9.2150
record was this comparison. Same shape as the output-layout bug: internally consistent,
externally wrong. `ceq_hybrid_attention` now calls transformers'
`sdpa_attention_forward`, so alpha=0 is bitwise **by construction** at every head grouping
and every version. Verified bitwise at 4 and 2 kv heads, logits and greedy generation both.

### Signedness at lam=0.10 is EMERGENT, not structural

Tier 3 is the property that justifies the module, and at the shipped `lam = 0.10` it is not
free. An entry is negative only when `softmax(w)_ij < lam*softmax(-w)_ij`. Over 20 draws at
`nn.Linear` initialization: negative fraction mean **3.91e-04 (cpu) / 4.99e-04 (cuda)**, max
**1.085e-03**, any-negative in 14/20 (cpu) and 19/20 (cuda) draws, at a mean within-row logit
spread of **0.677 / 0.631**. Against unit-variance logits (spread 3.256): **5.25e-02**, and
at 8x: **0.225**. The transition is sharp — at logit scale 0.25 (row spread 0.8141) it is
**0 of 4096**; at 0.50 (spread 1.6282) it is **19 of 4096**. `min A = -0.1080` on record is
the TRAINED 3.3M model. At `lam = 1.0` signedness IS structural — rows sum to exactly zero,
measured `|row sum| < 1e-6` — at the cost of annihilating the constant vector.

### Numbers this round corrected in shipped files

- `COSTS["content_conditional_sign_decay"]` shipped **0.10547 at s=16 and 0.03516 at s=32**
  in its first version. Both invented; DONE.md:975 and DONE.md:1104 independently record
  **0.08887** and **0.02637**. The guard checked the endpoints only and passed. It now
  asserts the whole dict and every rate verbatim against this log.
- A `COSTS` entry cited "gate recall 0.432 vs random 0.406 at s=256". Not in this log.
  Removed rather than kept.
- `CEQModel.forward` said gradient checkpointing was "1271.2 MiB -> under 0.6x" while `COSTS`
  in the same file said **1287.5 -> 487.7 MiB = 0.379x**. DONE.md:773 records the latter.
- `modeling_ceq.py` cited `test_tied_embeddings_survive_a_save_load_round_trip`. That test
  does not exist anywhere in the repository; the real ones are named now.
- `ceq/lm.py`'s `sgate` docstring gave `A = (rho/2)(softmax(w) - softmax(-w))`, the `lam = 1`
  special case, while the code has always had `lam` — the knob that got the operator from
  1.2099 to parity.
- `MODEL_CARD.md`'s only memory claim was the **1.31x-1.94x inference** figure. The training
  table (1.44x / 1.74x / 2.65x / 4.45x / 8.06x) is now in it.

### Triton reaches `sys.modules` through transformers, not through this package

`import torch` leaves `"triton" in sys.modules` **False**; `import transformers` **False**;
`from transformers import LlamaForCausalLM` **True**. Neither shipped file contains a triton
import. The smoke test's first version asserted `"triton" not in sys.modules` and was wrong.

### Gradient checkpointing was already wired

`gradient_checkpointing_enable()` sets `model.gradient_checkpointing = True` and
`_gradient_checkpointing_func`. Not a build claim of this round; measured and left alone.

### The stranger's command

```bash
python -m ceq.hf.smoke      # CPU only, no GPU, no Triton, no network
```

7/7 on python 3.11.9 / torch 2.5.1+cu121 / transformers 5.3.0, with `CUDA_VISIBLE_DEVICES=""`.
It prints the version matrix it ran on and the whole price list before the checks.

---

## Round 4 — CAMERON: the capability number, and it is a loss

The blocker that stopped round 2 (`lm.TinyLM` had a fixed 256-way head, COGS needs 874 word
types and sequences to 543) is gone — `lm.py` forwards model kwargs. COGS ran. Every finding
below is bound to a test in `tests/cameron/` that was RED before it was GREEN, logged to
`house-events.jsonl` by a `pytest_runtest_logreport` hook rather than transcribed, and every
test parametrizes over cpu and cuda.

### FINDING 1 — `sgate` scores 0.0000 on COGS-gen against a softmax control at 0.0293

`tests/cameron/test_capability_result.py`, 10 tests, RED on a missing artifact then GREEN.
Runner `python -m ceq.capability`; artifact `results/capability.json`.

3,652,096 parameters in BOTH arms — identical, not similar, because the operator carries no
parameters. Identical steps (3000), lr (3e-4), batch (32), seed (0), data, and eval
subsample. Decoder-only, from scratch, byte-free word-level vocabulary of 874, context 192,
greedy exact match on 512 generalization items.

| arm | COGS-gen | in-distribution (gate, 256 items) |
|---|---|---|
| softmax control | **0.0293** (15/512) | **0.9258** |
| `sgate` | **0.0000** (0/512) | **0.7734** |
| published, arXiv:2010.05465 Table 2 | 0.35 ± 0.06, 5 seeds, 21,000 items | 0.96 |

One-sided Fisher exact for softmax > sgate: **p = 2.8e-05**. This is a loss, not a tie.

**The deficit is already there in-distribution.** 0.7734 against 0.9258 on the split both
arms trained on. Whatever `sgate` costs, it costs it before generalization is tested — the
same shape as W10's val-loss gap that *widened* with budget.

**Architecture mismatch is stated, not hidden.** The published 0.35 is a 2+2-layer
encoder-decoder at 9.5M parameters. No COGS number exists for a decoder-only causal LM
trained from scratch (searched; arXiv:2310.19956 pretrains on 131B C4 tokens first). The
0.35 is a landmark, never a matched control; the matched control is the softmax arm in the
same table, and that arm learns the task — 0.9258 in-distribution against a published 0.96.

**Truncation ceiling reported with the number, not omitted.** `max_new = 192`, so the 4.88%
of scored items whose gold logical form runs longer are wrong whatever is emitted: the
achievable score is **0.9512**, identical for both arms.

### FINDING 2 — SCAN addprim_jump resolves nothing: both arms 0.0000

3 seeds, 4000 steps, 512 test items, 3,183,104 parameters both arms.

| arm | per seed | median |
|---|---|---|
| softmax | 0.0000 / 0.0000 / 0.0059 | **0.0000** |
| `sgate` | 0.0000 / 0.0000 / 0.0000 | **0.0000** |
| published vanilla Transformer, arXiv:2107.01366 Table 3 | — | 0.034 ± 0.020 **SEM** |
| published `+T5` relative-position attention, same table | — | 0.430 ± 0.095 SEM |

Confirmed by direct fetch: the caption reads "± denotes 1 SEM", numbers are percentages.

At n = 512 a zero carries a 95% Clopper-Pearson upper bound of **0.00583**, and the
published 0.034 would have shown ~17 solved items — so both arms genuinely miss the
published floor rather than being unresolved against it. But two arms at zero **separate
nothing from each other**, which is exactly the failure `tests/cameron/test_arc_reality.py`
was written to predict. The split with the nonzero floor still could not carry the
comparison at this budget.

### FINDING 3 — a reference this project cited for four rounds is a different task

Independently confirmed by direct fetch of arXiv:2112.00578. "Edge Transformer 0.874 ±
0.004 against a Universal Transformer control at 0.784" is Table 4 **top section**, captioned
"graph prediction accuracy", on "the graph representation of the COGS semantic parses
provided by Ontanón et al. (2021)". It is not the sequence-generation task. The 0.784 has no
± at all and is quoted from Ontanón et al. 2021, not run by those authors. The paper reports
no sequence-generation number of its own.

The only altered-attention precedent on sequence-generation COGS remains Csordás et al. 2021
Table 3: Trafo 0.80 ± 0.00, Rel. Trafo 0.81 ± 0.01, Uni. Trafo 0.78 ± 0.03, Rel. Uni. Trafo
0.77 ± 0.01. **About four points, not fifty-two.**

`arXiv:2605.20798` verified, with a caveat the citing text keeps dropping: the paper's own
words for ρ = −0.27 are that on seven improvers it is "a directional signal, not a precise
estimate" (p ≈ 0.56 two-sided). The 2–3%-loss / 6–16-CLIMB-point sentence is verbatim.

### SHIPPED — the diagnostic, one command, CPU, no GPU

`tests/cameron/test_diagnose_package.py`, **12 RED then 12 GREEN in 1176s**.

```bash
python -m ceq.diagnose            # ~2 min CPU;  --fast ~25 s;  --json for scripts
```

`ceq/diagnose.py` prints three things and refuses to print the first without the second:

1. `sign_flip_rate` over five operators at depth 1 and 2, with **softmax as the in-table
   control reading exactly 0.0**. One probe call at 128 draws costs **0.62 s** on CPU.
2. **The context-decay curve**, `s = 8…128`, with its fitted log-log slope. At 1024 draws it
   reproduces the published curve entry for entry — 0.17480 / 0.08887 / 0.02637 / 0.01172 /
   0.00391 — which required finding that the published sweep used `hops = 2`, the campaign's
   shipped point, not the probe's default `hops = 3`. `--fast` uses 512 draws, the smallest
   count whose slope still lands within 0.35 of −1.389 (measured: 128 → −1.295, 256 →
   −1.551, 512 → −1.402).
3. **The interventional code corpus with its scorer.** CPython is the oracle; no answer key
   exists in the repository. The predict-the-mean line at exactly 1.0 is printed next to the
   arms rather than left for the reader to remember.

`test_the_diagnostic_is_one_command_that_needs_no_gpu` runs it in a subprocess with
`CUDA_VISIBLE_DEVICES` emptied, so a hidden GPU dependency fails on this box rather than at
a reader's.

### What this round did NOT establish

One seed on COGS, 3000 steps, one context length, one learning rate. A 3× budget run
separates "slower to fit" from "cannot fit", and the in-distribution 0.7734 is the number
that would move first. `signmag` — which reads the highest sign-flip rate on the shipped
diagnostic, 0.1641 / 0.2969 at depth 1/2 against `sgate`'s 0.1484 / 0.0938 — has still never
been trained on either benchmark.

### FINDING 5 — the ratio HOLDS and the property DIES

The L1 operator's pathology (1.2191x at 250 steps -> 1.337x at 600) does NOT reproduce on
`sgate`. 600 steps, seed 0, lr 1e-3, measured by the tests themselves:

| axis | values | ratio |
|---|---|---|
| seq | 64 / 128 / 256 | 1.0356 / 1.0483 / **1.0340** |
| d_model | 128 / 256 / 384 | 1.0777 / 1.0312 / **1.0551** |

Both non-monotone, both inside the 1.0199-1.0413 five-seed spread already on record. GREEN:
`test_claim_the_parity_ratio_does_not_widen_with_context_length[cuda]` (164.9 s),
`test_claim_the_parity_ratio_does_not_widen_with_model_width[cuda]`.

**This is the worse result, not the better one.** Over exactly the seq 64 -> 256 range where
the ratio sits still at 1.034-1.048, the content-conditional sign rate falls 0.17480 ->
0.00391. The projected 300M outcome is therefore validation-loss parity with an operator that
has become indistinguishable from softmax -- bought at 1.32x wall-clock and 2.65x-8.06x
training memory. A widening ratio would at least have been a reason to stop.

### Infrastructure note — the budget axis crashed twice before it measured

`test_claim_the_parity_ratio_does_not_widen_with_training_budget[cuda]` (600/1200/2400 steps
x 2 seeds) died twice with `RuntimeError: CUDA error`, first as "an illegal memory access was
encountered" and then as "unspecified launch failure", surfacing at `ceq/lm.py:58` on a plain
`x.to(device)` and cascading into the other two slope tests through a poisoned context.
Attempt 2 began with the card idle at 99 MiB of 8188. A fresh process runs a trivial matmul
fine, and the two short slope tests pass; the card sat at 88-89 C under 100% load with up to
five concurrent CUDA processes from parallel sessions on the box. Treated as environmental.
NO ratio was produced by those crashed runs and none should be read from them.


---

## LOOP ITERATION 1 — 2026-08-25 — Phase 0. ARMS-DISTINCT bind written, RED first.

ACTION (one): wrote the ARMS-DISTINCT instrument, `tests/loop/test_arms_distinct.py`.

CALIBRATION [RUN] `python run_calib.py --self-test` -> exit 0.
The gate self-tested by rejecting a target off by 1e-9 (exit 1), then passed
4/4 bit-identical. NOTE: until today that script never compared anything --
it printed its targets as strings and always exited 0. It was the ELEVENTH
broken instrument here. Repaired before use; never trust a green from the
non-self-test path.

[RUN] `CUDA_VISIBLE_DEVICES="" python -m pytest tests/loop/test_arms_distinct.py -q`
Result: 5 passed, 1 failed in 83.39s. Output: results/iter01_arms_distinct.txt

RED-FIRST EVIDENCE (the instrument was seen to fail before any green trusted):
  * test_the_bind_fires_on_a_known_duplicate PASSED -- feeding it the same arm
    under two names raises AssertionError, reproducing the ParaFormer bug shape.
  * test_the_bind_does_not_fire_on_two_genuinely_different_arms PASSED -- it
    does not cry wolf.
  * test_the_structural_check_would_have_caught_the_historical_bug PASSED --
    on a verbatim reconstruction of the pre-fix dispatch it flags exactly
    ['paraformer'] and nothing else.

BEHAVIOURAL BIND [RUN]: GREEN on cpu. 8 arms (signed, softmax, sgate,
paraformer, signmag, deltanet, tgate, tgatex), 28 pairs, 0 collisions across
6 seeds at hops=2, s=16, n_draws=64. No arm reports another arm's number.

STRUCTURAL BIND [RUN]: **RED**. Reported missing: ['softmax'].
  Assessment: HALF-TRUE, and the instrument is at fault, not the arm.
  `softmax` has no literal in the dispatch because it is the `else`
  fallthrough. That is benign FOR SOFTMAX -- but the `else` is the exact
  mechanism of bug 9: any arm accepted by the guard clause and absent from the
  dispatch silently executes softmax and reports that number under its own
  name. The check cannot yet distinguish "documented default" from "silent
  fallthrough", so it over-fires on the former while still needing to catch the
  latter.
  Per loop governance a RED is overturned only by convicting the INSTRUMENT.
  Iteration 2 sharpens it: every arm except ONE explicitly-declared default
  must appear in the dispatch, and the default must be named in the test so a
  newly-added arm cannot inherit it silently.

CHECKLIST: no status changed. This is Phase-0 instrument work; no MANDATORY
item was tested, so no LOCK was taken.


---

## LOOP ITERATION 2 — 2026-08-25 — Phase 0. RED-1 resolved by convicting the instrument.

ACTION (one): repaired the structural half of the ARMS-DISTINCT bind.

CALIBRATION [RUN] `python run_calib.py --self-test` -> exit 0, 4/4 bit-identical
(signed 0.046875 / sgate-1 0.0234375 / sgate-2 0.1640625 / softmax 0.0). The
self-test rejected a 1e-9-off target first.

DIAGNOSIS [READ] `ceq/bench.py::sign_flip_rate`. `softmax` has no literal in the
dispatch because it is the `else` fallthrough in BOTH branches: the operator
selection ends `else: a = _softmax_operator(qq, kk, window=window)` and the
path-sum selection ends `else: h = a @ h` (single hop). Every other arm --
deltanet, tgate, tgatex, signed, sgate, signmag, paraformer -- appears
explicitly. So iteration 1's RED was a FALSE POSITIVE on the documented default,
while the underlying hazard it was built for is real and unchanged.

REPAIR: excuse exactly ONE declared default (`DEFAULT_ARM = "softmax"`), declared
in the test rather than inferred -- an inferred default would silently absorb
whichever arm happened to be last, which is the bug this file exists to catch.
The excusal is then fenced by three tests so it cannot become a hole:
  * test_excusing_the_default_does_not_excuse_a_missing_arm -- on a verbatim
    reconstruction of the pre-fix dispatch, with softmax excused, `paraformer`
    is STILL flagged. Detection of the real bug shape is provably intact.
  * test_a_newly_added_arm_cannot_inherit_the_default_silently -- a hypothetical
    unwired arm is flagged against the REAL dispatch.
  * test_the_declared_default_really_is_the_else_branch -- softmax is excused
    only because it IS the fallthrough, verified not assumed.

SECOND REPAIR, same action: the default-is-else check first used a fixed line
offset (`seg.split("
")[1]`). [RUN] it discriminated correctly on today's
formatting (True on good, False on bad) but would read the wrong line after any
reflow -- a check that depends on whitespace fails silently later. Replaced with
a scan for the first non-blank non-comment statement of each `else:` block, and
its RED-first case (`test_the_default_is_else_check_can_fail`) now lives in the
file permanently instead of in a shell transcript.

[RUN] `CUDA_VISIBLE_DEVICES="" python -m pytest tests/loop/test_arms_distinct.py -q`
Result: **9 passed in 61.96s**. Output: results/iter02_arms_distinct.txt

RED-1 CLOSED. Resolved per governance by convicting the INSTRUMENT; no arm was
adjusted and no assertion was weakened -- the repaired check is strictly sharper
than the original, since it now also catches an unwired NEW arm, which the
iteration-1 version could not distinguish from the default.

CHECKLIST: no status changed. Phase-0 instrument work; no MANDATORY item tested,
no LOCK taken.


---

## LOOP ITERATION 3 — 2026-08-25 — Phase 0. M2 pivot probe built, RED first.

ACTION (one): wrote `scale/pivot_probe.py`, the M2 instrument, with ARSENAL A1
(anti-concentration / Littlewood-Offord) decomposition built in rather than
bolted on.

CALIBRATION [RUN] `python run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

DESIGN [READ] `scale/pivot_probe.py`. out = v + Av + (A[:,P] A[P,:])v, |P| = k.
Causality is automatic and needs no extra mask: (A[:,P]A[P,:])_ij = sum_p A_ip
A_pj is nonzero only when j < p < i, because A is already strictly lower
triangular. Rank <= |P| by construction, which IS the mechanism.

ANTI-RIGGING, stated because the multizoom artifact was exactly this shape.
P is selected by CONTENT ONLY (key-norm), never using c. The `c in P` arm then
draws c from whichever tokens landed in P; `c not in P` draws from the
complement. Conditional measurement, not a rigged one. The remaining hole is
covered by the pre-registered kill: if `c not in P` is ALSO flat, the mechanism
story is false even if the pivot number looks good.

PROTOCOL: SCALING (i=s-1, j=s/4). Printed by the probe in its own output, per
the geometry-trap rule. A1's first attempt was VOID for running PINNED.

RED-FIRST EVIDENCE -- the instrument was shown to fire before any pivot number
was taken. [RUN] `python scale/pivot_probe.py --sizes 32 128 --n 128`:

  arm             s=32                  s=128                 slope
  dense_signed    0.0234[0.005,0.067]   0.0078[0.000,0.043]   **-0.792**
  sgate           0.0312[0.009,0.078]   0.0000[0.000,0.028]   nan (1/2 nonzero)
  random          0.4375[0.350,0.528]   0.4844[0.395,0.574]   +0.073
  dense_unsigned  0.0000[0.000,0.028]   0.0000[0.000,0.028]   nan (0/2 nonzero)
  pivot_unsigned  0.0000[0.000,0.028]   0.0000[0.000,0.028]   nan (0/2 nonzero)

  * dense_signed -0.792 FIRES the pre-registered kill (< -0.3) on an arm already
    known to decay. The instrument can go RED. This is the RED-first bind.
  * sgate collapsing to exactly 0.0000 at s=128 reproduces its recorded death.
  * BOTH ENDS OF THE INSTRUMENT ARE NOW VERIFIED, which no previous probe here
    had: a FLOOR and a CEILING.

FLOOR [RUN + DERIVED]: softmax reads exactly 0.0000. This is a THEOREM, not a
suspicious number, and therefore satisfies the probe's own "an exact 0.0000
means broken until a known-truth synthetic passes" rule. With out = v + Av +
A^2 v the influence Jacobian is I + A + A^2; for non-negative A that is
non-negative entrywise, so a sign flip is structurally impossible. Softmax
cannot register here for the same reason it cannot represent negation.

CEILING [RUN], and an instrument caveat recorded rather than glossed: the
`random` arm draws a FRESH matrix per intervention branch, so its ~0.48 is a
NOISE CEILING (two independent matrices disagree in sign about half the time),
NOT a chance baseline for content-conditional sign. The true null for this
property is ZERO -- an operator that ignores content cannot flip. Anyone reading
0.5 as "chance" would mis-set every future threshold. The floor control above is
the correct null; `random` is a liveness check that the probe is not stuck at 0.

CHECKLIST: no status changed. Phase-0 instrument work. M2 itself is UNTESTED --
the probe now EXISTS and has been RED once, which is the Phase-0 requirement;
M2's actual test (s=8..2048, 16384 draws at the tail, three placements) is
Phase 1 and will take the M2 LOCK when it runs.

NOTE: the house-mode board process exited 127 (`house.py` launcher). The board
is a read-only mirror and nothing depends on it; not chased.


---

## LOOP ITERATION 4 — 2026-08-25 — Phase 0 COMPLETE. M3 capability probe built, bar calibrated.

ACTION (one): wrote `scale/negation_scope.py`, the M3 instrument, and calibrated
its absolute bar at THREE points before any arm may be credited with beating it.

CALIBRATION [RUN] `python run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

THE TASK [READ] `scale/negation_scope.py`. y = payload_value * flipper_sign,
with payload and flipper at DIFFERENT positions, so the task cannot be solved by
retrieving one negative value -- it requires combining two positions
multiplicatively, which is the shape of "token B suppresses token A". Flipper at
distance d from the query; payload adjacent to the query, so that a failure
cannot be a RETRIEVAL failure wearing a sign failure's name.

ORACLE IS EXECUTABLE, NO ANSWER KEY ON DISK. `oracle(x, f, p)` recomputes y from
x. The W4 death was a corpus split by literal value, which measured embedding
coverage rather than generalization; a stored answer key is the same failure with
a shorter fuse.

BAR CALIBRATION [RUN] `python scale/negation_scope.py --s 512 --distances 256
--n 4096` -> exit 0. Output: results/iter04_m3_bar_calibration.txt

  predictor            NRMSE        check
  predict_the_mean     1.000000     exactly 1.0 -- this IS the definition of the bar
  payload_only         1.414204     FAILS the bar, and see below
  oracle               0.000000     a perfect predictor reads 0, so the task is solvable

  BAR CALIBRATED.

THE MIDDLE POINT HAS A CLOSED FORM, which is a stronger check than was designed
for [DERIVED, confirmed by RUN]. The flipper-blind predictor outputs `payload`
when the truth is `payload * sign`. Its error is `payload*(1 - sign)`, which is 0
or 2*payload with equal probability, so E[err^2] = 2*var(y) and NRMSE = sqrt(2) =
1.4142136. Measured **1.414204**, agreeing to 1e-5. The bar is therefore pinned
at BOTH ENDS AND IN THE MIDDLE by an analytic value -- no previous instrument in
this project had a derivable interior calibration point.

WHY THIS ITEM MATTERS MOST: M1 is a PRECONDITION (SignGT/SDA/Cog all pass it);
M2 is a STATISTIC, and CHECKLIST.md's preamble says statistics count for nothing.
M3 is the only MANDATORY item that is a CAPABILITY.

PHASE 0 IS NOW COMPLETE: calibration green (gate repaired first), ARMS-DISTINCT
9/9, M2 probe RED-first proven at slope -0.792, M3 bar calibrated at 3 points.

CHECKLIST: no status changed. The M3 PROBE exists and its bar has been shown to
fire; M3 ITSELF is UNTESTED and takes its LOCK when arms run in Phase 2.

---

### Landed while iteration 4 ran — two agent reports, both bearing on M3

**CAMERON [RUN, artifact results/capability.json, bound by
tests/cameron/test_capability_result.py]** -- a capability probe with a published
third column already exists, and the dense signed operator LOSES it:

  COGS generalization, exact match, 512 items, **3,652,096 parameters in BOTH
  arms**, identical steps/lr/batch/seed/data, one seed:
    softmax control  **0.0293** (15/512)   in-distribution **0.9258**
    sgate            **0.0000** (0/512)    in-distribution **0.7734**
    published enc-dec arXiv:2010.05465 Tab.2  0.35 +/- 0.06
  One-sided Fisher exact, softmax > sgate: **p = 2.8e-05**. A loss, not a tie.
  The deficit is already present IN-DISTRIBUTION, on the split both arms trained
  on, so it is NOT a generalization story.

  This does not test M3 -- M3 is about the PIVOT arm and this is the DENSE signed
  arm -- but it is strong prior evidence that dense signedness costs capability,
  and it is the number the pivot arm must beat to mean anything.

**CHASE [RUN, tests/chase/test_hub_package_hardening.py, 58 passed cpu+cuda]**:
  * The shipped package implemented the WRONG operator: `rho*w/||w||_1` at
    rho=0.9 hops=3, while the 1.0334 median is sgate at rho=1.5 lam=0.10 hops=2
    -- relative difference **1.818e+00**, on record since round 3, never acted on.
  * **"alpha=0 is BITWISE stock" was FALSE.** It had been asserted against this
    repo's own `ceq.hybrid.stock_attention`. Against the model's real sdpa path:
    0.000e+00 at 4 kv heads but **1.490e-07 max abs / 2.609e-07 relative at 2 kv
    heads**, because Qwen2.5-0.5B is GQA and `_repeat_kv` materializes what
    transformers hands torch as `enable_gqa=True`. The 9.2149-vs-9.2150 record
    was that flawed comparison. Same failure class as the parity bug: comparing
    against our own reimplementation instead of the real path.
  * **COSTS shipped two INVENTED numbers** -- 0.10547 at s=16 and 0.03516 at
    s=32 against the log's 0.08887 and 0.02637. His own guard checked only the
    endpoints and passed them.
  * **Signedness at lam=0.10 is EMERGENT, not structural**: negative fraction at
    `nn.Linear` init is 3.91e-04 (cpu) / 4.99e-04 (cuda). The `min A = -0.1080`
    on record is the TRAINED model. Structural only at lam=1.


---

## LOOP ITERATION 5 — 2026-08-25 — HEALTH INSPECTOR PASS (mandated every 5th iteration).

ACTION (one): the Inspector pass. Four checks, all executed, plus one finding
against this loop's OWN earlier work.

**CHECK 1 — calibration.** [RUN] `python run_calib.py --self-test` -> exit 0,
4/4 bit-identical. The gate self-tested by rejecting a 1e-9-off target first.
CLEAN.

**CHECK 2 — LOCK hashes.** ASSERTED, not assumed: `grep "^LOCK " STATE.md`
returns nothing, which is correct because no MANDATORY item has been tested yet.
An Inspector who assumed "no locks, nothing to check" would pass a tampered file.
CLEAN.

**CHECK 3 — re-run one published number, bit-identical.** Published numbers in
record order: 0 signed-0.046875, 1 sgate2-0.1640625, 2 dense_signed-slope-0.792,
3 M3-bar-1.414204, 4 Zaslavsky-tgate-0.5056, 5 Carnot-Q/topo, 6 ARMS-DISTINCT-9/9.
Selector: iteration 5 mod 7 = **5** -> the Carnot ratio. [RUN] reproduced exactly
at every s: 1.67 / 3.00 / 5.67 / 21.67 / 85.67 / 171.00 against (s+1)/3, all OK.
CLEAN.

**CHECK 4 — does any doc still ASSERT a number that has been corrected?**
Searched for the dead ParaFormer figures (0.00146, 84x, 84x) and Chase's two
invented COSTS figures (0.10547, 0.03516).
Every hit is a RETRACTION or a corrected value carrying its provenance:
  * README.md:49-51 "so the 84x was softmax's zero. Corrected, the separation
    is 5.6x"; :362 "Round 4 retracts the 84x"; :385 "5.6x, not 84x"
  * MODEL_CARD.md:208 records it as "a softmax [number]"
  * ceq/hf/modeling_ceq.py:127-131 -- the dict holds the CORRECT 0.08887 and
    0.02637, with a comment naming the invented 0.10547/0.03516 and stating that
    the old guard only checked endpoints.
  * README.md:141 `0.03516` is a FALSE MATCH -- it belongs to the floor-on/off
    control at 256 draws with j=1, a different sweep. The decay table at :133
    carries the correct series. Verified by reading the lines, not by assuming.
CLEAN -- no strikes.

**FINDING — against this loop's own iteration 3.** `scale/pivot_probe.py`
defines six arms (pivot_signed, dense_signed, pivot_unsigned, dense_unsigned,
deltanet, sgate, random) and **NO ARMS-DISTINCT bind covers any of them**. The
iteration-2 bind covers `ceq/bench.py::sign_flip_rate` only. That is precisely
the ParaFormer hazard (G3) reopened in a new file: an arm whose branch is
mis-wired would report another arm's number under its own name, and nothing
would catch it.

  Not a false alarm: the arms ARE structurally distinct today (`pivot_hop2(a, P)`
  vs `a @ a`), but nothing ASSERTS it, and "correct today, unasserted" is the
  state every one of the thirteen broken instruments was in.

  The repair needs care, and the naive version would be instrument fourteen:
  `dense_unsigned` and `pivot_unsigned` BOTH read exactly 0.0000, so a
  behavioural fingerprint bind would flag them as duplicates. They are not
  duplicates -- both are non-negative, so `I + A + A^2` is non-negative
  entrywise and neither can flip, by theorem. The bind for this probe must
  compare the raw hop-2 TENSORS, not the rates, and must carry the same
  known-truth exemption the M3 bar carries.

CHECKLIST: no status changed. Inspector passes never change item status; the
Inspector audits the log and never rules on whether a finding is correct.


---

## LOOP ITERATION 6 — 2026-08-25 — pivot-probe arms bound. Instrument 14 caught in the act.

ACTION (one): closed the iteration-5 Inspector finding by binding
`scale/pivot_probe.py`'s seven arms.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

REFACTOR + DRIFT CHECK. Extracted `build_arm()` as the single source of truth for
arm construction, so the bind exercises the SAME code path the probe runs. A test
that rebuilds the operators itself proves only that two reimplementations agree --
which is how this project published 89,400.180 against 1.667, and how "alpha=0 is
bitwise stock" survived until checked against the real sdpa path.
[RUN] iteration 3's numbers after the refactor: dense_signed 0.0234 / 0.0078
slope **-0.792**, sgate 0.0312 / 0.0000, random 0.4375 / 0.4844 slope **+0.073**
-- BIT-IDENTICAL to the published values. No number moved.

**INSTRUMENT 14, caught by its own test.** The first run of
`test_pivot_routing_actually_restricts_rank` reported **rank 37 for a matrix that
is provably rank <= 8**, since `(64x8)(8x64)` cannot exceed 8. Investigated
rather than patched. Singular values:

    9.090e-01 5.571e-01 1.952e-01 1.676e-01 1.087e-01 9.557e-02 4.158e-02
    2.166e-02 | 1.734e-08 1.671e-08 1.081e-08 7.651e-09 ...

Eight real values, then a cliff into float32 rounding noise. Cause: the matrix is
COMPUTED in float32 and was cast to `.double()` afterwards.
**Casting to double after the fact does not recover precision -- it promotes the
float32 noise to double, and `matrix_rank`'s default tolerance then counts 29
noise directions as real rank.** The MECHANISM was never wrong; the measurement
was. Fixed with an explicit tolerance `max(dims) * eps(float32) * sigma_max`, and
the test now also asserts the spectral gap so the rank claim cannot be an
artefact of the tolerance itself.

RESULT [RUN] `pytest tests/loop/test_pivot_arms_distinct.py -q` -> **6 passed**,
cpu and cuda. Output: results/iter06_pivot_arms_distinct.txt

  * ARMS-DISTINCT: 7 arms, 21 pairs, **0 collisions**, on cpu AND cuda.
  * **rank(pivot hop2) = 8 == |P|; rank(dense hop2) = 54; spectral gap
    sigma_8/sigma_9 = 1.249e+06.** The mechanism claim -- that routing restricts
    the two-hop operator to rank |P|, so one token's share becomes 1/k
    independent of s -- is now MEASURED, not assumed. This is the first positive
    structural result for the pivot design.
  * `test_the_rate_level_bind_would_have_false_positived` pins why this file
    compares TENSORS: `dense_unsigned` and `pivot_unsigned` both read exactly
    0.0000 as a rate (both non-negative, so `I + A + A^2` cannot flip, by
    theorem) while their hop-2 tensors differ. A rate fingerprint would have
    called them duplicates.

CHECKLIST: no status changed. Instrument work. M2 remains UNTESTED and takes its
LOCK in Phase 1.

---

### CAMERON's full report — two CITATION corrections against my own earlier prose

**The COGS reference in the brief was wrong, and it was mine.** "Edge Transformer
0.874 +/- 0.004 against a UT control at 0.784" is the **graph-prediction**
reformulation (arXiv:2112.00578 Table 4, captioned "graph prediction accuracy"),
NOT sequence-generation COGS, and the 0.784 has no error bar. The only
sequence-generation altered-attention precedent is **Csordas 2021, 0.81 +/- 0.01
vs 0.80 +/- 0.00 -- four points, not fifty-two.** Confirmed by direct fetch.

**arXiv:2605.20798's rho = -0.27 was cited above its strength.** The paper itself
describes it as "a directional signal, not a precise estimate" (n = 7,
p ~ 0.56). The verbatim 2-3%/6-16-CLIMB sentence IS verified; the correlation is
not evidence of the strength previously attached to it.

**`signmag` has the HIGHEST sign-flip rate on the shipped diagnostic**
(0.1641/0.2969 vs sgate 0.1484/0.0938) and has still never been trained on either
benchmark.

SCAN addprim_jump: softmax median 0.0000, sgate median 0.0000, published vanilla
Transformer 0.034 +/- 0.020 SEM. **Two arms at zero separate nothing.**


---

## LOOP ITERATION 7 — 2026-08-25 — PHASE 1 BEGINS. M2 LOCKED and launched.

ACTION (one): took the M2 LOCK and started the M2 run.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

**LOCK M2 efadc390c93f** — sha256[:12] of M2's item text in CHECKLIST.md (592
chars), recorded in STATE.md and archived verbatim at results/m2_item_text.txt.
**M2's item text is now FROZEN and may never be edited.** This is the first
MANDATORY item to be tested; the Health Inspector re-verifies this hash on every
5th iteration and a mismatch is tampering.

TIMED BEFORE BUDGETED, so the draw count is stated rather than silently capped
[RUN]:

    s=8  20.2 ms/draw | s=128  6.2 | s=512  14.4 | s=1024  33.0 | s=2048 124.5
    16384 draws at s=2048 costs 0.57 h -- affordable, so the tail gets the full
    count the item asks for rather than a quiet truncation.

BUDGET, STATED: 4096 draws at s <= 512, **16384 at s >= 1024**. That is what
"16384 draws at the tail" means -- the tail is where rates are small and the
Clopper-Pearson interval needs the draws.

SIX CELLS, so the claim, its own control, and BOTH instrument ends are in one
table:
  pivot_signed c IN P      <- the claim
  pivot_signed c NOT in P  <- the control that can falsify the MECHANISM
  pivot_signed c windowed
  dense_signed             <- the known-decaying reference (-0.792 on record)
  softmax FLOOR            <- structural 0.0000, by theorem
  random CEILING           <- liveness, ~0.48

PROTOCOL: SCALING, printed by the runner itself.

LAUNCH NOTE, recorded because it nearly produced a phantom result: the first
launch used `nohup ... &`, and the process did not survive the shell -- the
output file was 0 bytes while 7 unrelated python processes were alive, so a
casual `pgrep`-style check would have read as "running". Relaunched through the
harness background mechanism. **A run that is not producing output is not
running, whatever the process table says.**

RESULT: pending. Recorded next iteration, with the frozen kill applied.

CHECKLIST: M2 status stays UNTESTED until the run returns; the LOCK is taken at
the START of the test, per governance, not at its end.


---

## LOOP ITERATION 8 — 2026-08-25 — prior-art fetch: A1's [U] citations cleared.

M2 is still running, and LOOP_PROMPT.md forbids starting anything that would
compete with it. Waiting is not an action; a PRIOR-ART FETCH is a legal one and
touches nothing the run uses. A1 attaches to M2, so if M2 goes RED the ARSENAL
usage rule freezes A1 -- fetching now is the last cheap moment to do it.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): verified A1's two load-bearing citations BY DIRECT FETCH, and
upgraded them [U] -> [V] in ARSENAL.md.

**[V] Tikhomirov**, *Singularity of random Bernoulli matrices*, **Annals of
Mathematics 191(2), 2020, pp. 593-634**, arXiv:1812.09016. For an n x n matrix
with independent +-1 entries, P(singular) = **(1/2 + o_n(1))^n**. Venue, year and
result all as stated in the arsenal. CONFIRMED.

**[V] Inverse Littlewood-Offord.** Posed by **Tao-Vu**, *Inverse
Littlewood-Offord theorems and the condition number of random discrete matrices*,
**Annals 169(2)**: large concentration implies most coordinates lie in a
**generalized arithmetic progression of small rank and small volume**, proved via
Freiman-type additive combinatorics. Sharpened to an **optimal** form (error term
removed) by **Nguyen-Vu**, arXiv:1004.3967. CONFIRMED.

So A1's design principle -- *anti-concentration stays large iff the weights carry
additive structure* -- is stated correctly and is attributable. It is the only
principled candidate on the board for arresting the decay, and it is no longer
resting on memory.

**[U] REMAINING, recorded rather than quietly dropped:**
Campos-Jenssen-Michelen-Sahasrabudhe (symmetric-matrix case) is still unfetched.
Not load-bearing for A1's use here; must be fetched before it is cited.

M2 INTERIM, **NOT A RESULT** -- recorded only so the next iteration knows the run
is alive and producing output:

    pivot_signed c IN P:  s=8 0.02466 | s=32 0.02856 | s=128 0.03101 | s=512 0.02881
    (4096 draws each)

Flat so far. **This decides nothing.** The `c NOT in P` control has not run, and
per the frozen kill a flat `c in P` with a flat `c NOT in P` means the mechanism
story is FALSE even though the pivot number looks good. Reading these four cells
as a win would be the exact error the kill was written to prevent.

CHECKLIST: no status changed. M2 remains UNTESTED until its run completes.


---

## LOOP ITERATION 9 — 2026-08-25 — prior-art fetch: A2 cleared, and it exposes a hole in M6.

M2 still running (no new cell since iteration 8; it is in the s=1024 tail at
16384 draws). Legal non-competing action taken: prior-art fetch.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

ACTION (one): verified A2's two load-bearing citations by direct fetch and
checked, by RUN, whether the chain applies to the operator that actually ships.

**[V] Berger power inequality** `w(A^m) <= w(A)^m`. Conjectured by Halmos, proved
by Berger, elementary proof by **Pearcy (1966)**. As stated in the arsenal.

**[V] Crouzeix-Palencia (2017)**: the numerical range is a **(1+sqrt2)-spectral
set**, `||f(A)|| <= (1+sqrt2) sup_{W(A)}|f|`. Crouzeix conjectures the constant
may be 2; 2 is best possible. **NEW and unverified:** arXiv:2608.03841 (2026),
"A solution to Crouzeix's conjecture" -- fetch before relying on constant 2.

**[RUN] THE CHAIN APPLIES -- AND IS VACUOUS AT THE CURRENT OPERATING POINT.**
On `tgate` (s=64, seed 0):

    w(A) = 1.499315   ||A|| = 2.837508   ||A|| <= 2w(A): TRUE
    h=1..4: ||A^h|| = 2.838e0 / 1.777e0 / 7.119e-1 / 2.099e-1
            2w(A)^h = 2.999e0 / 4.496e0 / 6.741e0 / 1.011e1     all hold

The inequality holds at every h. But **w(A) = 1.4993 > 1**, so the BOUND GROWS
with h while the true norm FALLS (the operator is nilpotent). A certificate whose
bound grows while the quantity it bounds shrinks certifies nothing.

**Consequence for M6, recorded now so it is not discovered in Phase 3:** M6's
guard is only a real constraint if it actually binds `w(A) <= 1`, and the shipped
operator does NOT satisfy that today. Enforcing it is a genuine change to the
operator, not a formality, and M6's own kill -- "training diverges under the
constraint at every lr" -- is a live possibility.

**Refinement to M6's statement.** `||A^h|| <= 2w(A)^h` needs only Berger plus the
standard `||M|| <= 2w(M)`; Crouzeix-Palencia is not required for it.
Crouzeix-Palencia buys the strictly better object: a bound on the WHOLE path sum
`p(A) = sum_h A^h`, which is what the operator computes, instead of a
term-by-term bound that is then summed. M6 should be stated on `p(A)`.
**This does not edit M6** -- M6 is UNTESTED and unlocked, but its text is the
user's and stays as written; this is a note on how to satisfy it.

CHECKLIST: no status changed. M2 remains UNTESTED and running.


---

## LOOP ITERATION 10 — 2026-08-25 — HEALTH INSPECTOR PASS. 3 clean, 1 failed, repair executed.
## Includes a CORRECTION to iteration 7's recorded lesson, which was wrong.

**CHECK 1 — calibration.** [RUN] `run_calib.py --self-test` -> exit 0, 4/4
bit-identical. CLEAN.

**CHECK 2 — LOCK verification. First pass where this does real work.**
`LOCK M2 efadc390c93f` checked THREE ways, not one:
  * sha256[:12] of M2's item text as it stands in CHECKLIST.md **now**: efadc390c93f  MATCH
  * sha256[:12] of the archived copy at results/m2_item_text.txt:      efadc390c93f  MATCH
  * archived text == live text, byte for byte:                          True
M2's frozen text has not moved since its test began. No tampering. CLEAN.

**CHECK 3 — re-run a published number.** Selector: iteration 10 mod 7 = **3** ->
the M3 bar. [RUN] reproduced bit-identical: predict_the_mean **1.000000**,
payload_only **1.414204**, oracle **0.000000**. CLEAN.

**CHECK 4 — M2 liveness. FAILED.** No cell had advanced since iteration 8: still
showing s=512 as the last completed cell across two full iterations, where
s=1024 at 16384 draws should cost about 9 minutes at the measured 33 ms/draw.
Per governance, a failed check REPLACES this iteration's plan with the repair.

**DIAGNOSIS [RUN].** `Get-CimInstance Win32_Process` showed **TWO** M2 runs alive
and competing:
    PID 17096  09:34:30  `python scale/run_m2.py`      <- the nohup launch
    PID 632    09:34:51  `python -u scale/run_m2.py`   <- the harness launch
Both were burning ~900 CPU-seconds at ~480 MB, both writing results/m2_run.txt,
duplicating every cell. Eight python processes were on the box in total (three
pytest runs, one at 9818 CPU-seconds, plus ceq.diagnose), so the duplicate was
compounding real contention.

**CORRECTION TO ITERATION 7 — the lesson recorded there is WRONG.**
Iteration 7 recorded: *"the first launch used `nohup ... &`, and the process did
not survive the shell"*, and drew the lesson *"a run that is not producing output
is not running."* **Both are false.** The nohup process survived perfectly well
for 30+ minutes; its output file was empty because that invocation had **no
`-u`** and Python was buffering. The 0-byte file was BUFFERING, not death.

The corrected lesson is the opposite of the one recorded, and sharper:
**an empty output file does not mean a dead process -- check the process table
before relaunching, or you will start a second copy of an expensive run.** The
iteration-7 entry is left in place (DONE.md is append-only and recorded results
are never edited) and is superseded by this paragraph.

**REPAIR [RUN].** Killed PID 17096, the orphaned duplicate. Verified afterwards
that exactly one `run_m2` process remains: PID 632, the harness-tracked run with
`-u`. The Inspector's own work was not touched -- its three pytest processes were
left alone.

CHECKLIST: no status changed. Inspector passes never change item status. M2
remains UNTESTED and now has the machine to itself for its own work.


---

## LOOP ITERATION 11 — 2026-08-25 — prior-art fetch: A3 cleared. M2 confirmed computing.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

**M2 LIVENESS, measured rather than assumed [RUN].** Still showing s=512 as the
last completed cell, a third iteration without a new line. Rather than repeat
iteration 7's mistake of inferring death from silence, measured the CPU:

    PID 632  CPU 1187.19 s -> 1239.14 s over 20 s wall  = 51.95 CPU-s / 20 wall-s
    ~2.6 cores busy, working set 570 MB      VERDICT: COMPUTING, not hung

Exactly one `run_m2` process remains after iteration 10's repair.

**INSTRUMENT NOTE: the 4-draw timing was an UNDERESTIMATE.** Iteration 7 measured
33.0 ms/draw at s=1024 from a 4-draw sample and budgeted from it. The tail cell
is plainly costing far more than the 540 s that predicted. A 4-draw sample is too
small to characterise a cell that runs 16384 draws -- warmup and cache effects
dominate it. Future budgets should time at least ~100 draws, or report the
estimate as a lower bound. The BUDGET itself is unaffected (draw counts were
stated in advance and are not being cut); only the ETA was wrong.

ACTION (one): verified A3's citations by direct fetch and upgraded ARSENAL.

**[V] Variation diminishing property** — first studied by **Schoenberg (1930)**;
total positivity was introduced by Schoenberg *in his study of
variation-diminishing kernels*, generalized by Schoenberg and **Karlin**.
Statement: a TP kernel applied to a function with n sign changes yields at most
n sign changes. **A totally nonnegative kernel can only DESTROY sign changes,
never create them** — the exact classical form of this repo's semiring theorem,
which is what makes M1 theorem-shaped rather than a benchmark delta.

**[V] Gantmacher-Krein (1950)** relating total positivity to variation diminution.
Newly surfaced, was not in the arsenal.

**[V] STRONGER THAN THE ARSENAL CLAIMED:** variation-diminishing *together with*
sign non-reversal **characterize** totally nonnegative matrices (arXiv:2103.05624,
arXiv:2007.09999). So "non-negative <=> cannot create a sign change" is an IFF,
not merely an implication. That is a better statement than A3 was resting on.

**[V] Postnikov**, *Total positivity, Grassmannians, and networks* — positroid
stratification of Gr_>=0(k,n), cells cut out by vanishing Plucker coordinates,
a special case of Lusztig's theory.

**[U] NOT FOUND and left unverified:** Karp's sign-variation characterization and
Galashin-Karp-Lam. Searched, not located. Recorded as unfetched rather than
quietly dropped; must not be cited until found.

STANDING CAVEAT, unchanged: A3 makes M1 theorem-shaped, but **M1 is a
PRECONDITION, not novelty** — SignGT, SDA and Cog Attention all pass it. The
novelty, if any, lives in M2 x M3.

CHECKLIST: no status changed. M2 remains UNTESTED and running.


---

## LOOP ITERATION 12 — 2026-08-25 — A4 cleared. TIER 1 OF THE ARSENAL IS NOW FULLY VERIFIED.

CALIBRATION [RUN] `run_calib.py --self-test` -> exit 0, 4/4 bit-identical.

**M2 ADVANCED.** s=1024 landed: **0.02966**, full 16384 draws, 471 s -- close to
the 540 s the original estimate predicted. So iteration 11's "the 4-draw timing
underestimated" note was itself too harsh: the budget was about right, and the
apparent stall was the DUPLICATE RUN contention that iteration 10 removed. Two
corrections in two iterations, both toward "measure, do not infer".

    pivot_signed c IN P: 0.02466 / 0.02856 / 0.03101 / 0.02881 / 0.02966
                         at s = 8 / 32 / 128 / 512 / 1024

Flat across a 128x context growth. **Still not a result** -- the `c NOT in P`
control has not run, and a flat claim beside a flat control is a RED.

ACTION (one): verified A4's citations by direct fetch.

**[V] Dauvergne, Ortmann & Virag**, *The directed landscape*, arXiv:1812.00309,
**Acta Mathematica 229(2), December 2022**. Airy sheet constructed and
characterized via the Airy line ensemble; last-passage geodesics converge to
random functions with **Holder-2/3-continuous** paths; completes the construction
of the central object of the KPZ universality class. As stated in the arsenal.

**[V] KPZ exponents**: fluctuation **1/3** with **Tracy-Widom** limit, transversal
exponent **2/3**; scaling relation `chi = 2 xi - 1` (arXiv:1211.0992).

**[U] Lee-Yang / partition-function zeros** -- not fetched, not to be cited.

**CAVEAT THAT WEAKENS A4, recorded because it is load-bearing.** Rigorous proofs
of the 1/3 and 2/3 exponents are **scarce outside a few integrable cases**. A4
predicts trained path weights localize on `O(s^{2/3})` corridors -- that
extrapolates an exponent proved for integrable models to a LEARNED,
non-integrable operator. **A4 SUGGESTS the measurement; it does not PREDICT it.**
Recorded at that strength and no higher.

**A4's kill is still owed, and its one attempt was VOID**: the participation
ratio was measured in the PINNED geometry (intermediate count ~5 regardless of
s), read a constant ~1.27-2.89, and meant nothing. Must be re-run under
PROTOCOL: SCALING.

**MILESTONE: Tier 1 of the arsenal (A1, A2, A3, A4) is now fully fetched and
verified**, over iterations 8, 9, 11, 12 -- all done as legal non-competing work
while M2 ran, at zero cost to the measurement. Three of the four came back with
something the arsenal did not have:
  * A2 -- the certificate is VACUOUS at the current operating point (w(A)=1.4993
    > 1, so the bound grows while the true norm falls). M6 must bind w(A) <= 1.
  * A3 -- variation-diminishing + sign non-reversal CHARACTERIZE TNN matrices.
    An iff, stronger than claimed.
  * A4 -- the KPZ exponents are proved only in integrable cases. Weaker than
    claimed.

CHECKLIST: no status changed. M2 remains UNTESTED and running (1 of 6 rows done).


---

## LOOP ITERATION 13 — 2026-08-25 — M2 DIED SILENTLY. Root-caused, and the architecture fixed. ADR-001.

**M2 DID NOT COMPLETE.** The harness reported **exit code 0**; the run had one
row of six, no `results/m2.json`, and the process was gone. Investigated instead
of trusting the exit code.

**ROOT CAUSE — three failures compounding, none of them OOM:**

1. **My own timeout killed it.** The backgrounded Bash call carried
   `timeout: 600000` (10 min, the tool's MAXIMUM). The run died at ~600 s, which
   is exactly when s=1024 finished: 117 s of small cells + 471 s = 588 s.
   **Any measurement over 10 minutes CANNOT complete in one harness call.** This
   is a hard platform constraint, discovered only by hitting it.
2. **`python -u ... | tee f` reports TEE's exit status, not python's.** A killed
   process was logged as success. A truncated run presented as a complete one --
   the exact shape of the thirteen instrument failures already on record.
3. **All state was in memory.** `results/m2.json` is written only at the end, so
   588 s of correct computation -- including a full 16 384-draw cell -- was
   destroyed rather than banked.

**OOM WAS RULED OUT BY MEASUREMENT, not assumed** [RUN]: s=2048 runs at
**63.1 ms/draw with RSS 596 MB** (512 -> 456 MB, 1024 -> 478 MB, 2048 -> 596 MB).
Free RAM was 2.8 of 15.7 GB, which made OOM the obvious hypothesis -- and it was
wrong. Checked before acting on it.

ACTION (one): **ADR-001** plus `scale/bucket.py`, restructuring every long
measurement as idempotent units journalled append-only and executed in
wall-clock-bounded buckets, one bucket per loop iteration.

Engineering tactics adopted, each against a hazard OBSERVED in this run:

  bucketing under the platform cap        <- the 600 s kill
  append-only journal, one line per unit  <- losing 588 s of correct work
  idempotent units keyed by full params   <- recompute drift, silent skips
  never pipe through `tee`                <- exit 0 from a killed process
  lock file with PID                      <- the nohup orphan doubling the work
  replay assertion, bitwise               <- undetected nondeterminism
  done/total accounting, refuse partial   <- truncated run reported as complete
  budget from >=100-draw timing           <- the 4-draw sample

The replay assertion is the part that pays twice: resume was going to be a leap
of faith, and instead it is a **free determinism audit** -- every resumed bucket
recomputes an already-journalled unit and requires a bitwise match.

SELF-TEST [RUN] `scale/bucket.py`: journals 3/3, second call skips all and
replay-verifies (`replay demo/s1: MATCH`), aggregates correctly, and **REFUSES**
to aggregate when one unit is missing.

CHECKLIST: no status changed. **M2 remains UNTESTED and its data is NOT lost --
it was never written.** The five completed `c IN P` cells (0.02466 / 0.02856 /
0.03101 / 0.02881 / 0.02966) are on record in this file but must be RECOMPUTED
through the journal, not seeded from prose. The LOCK stands: efadc390c93f.


---

## LOOP ITERATION 14 — 2026-08-25 — ADR-001 VALIDATED under a real kill. INSPECTOR: 4 STRIKES.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): wrote scale/m2_units.py (36 units) and ran bucket 1.

**THE BUCKET WAS KILLED AT 600 s AGAIN -- AND THIS TIME NOTHING WAS LOST.** The
harness cap fired (exit 143) mid-unit. The journal held:

    pivot_signed__in_P/s8     0.02466  n=4096    (4.24s)
    pivot_signed__in_P/s32    0.02856  n=4096    (4.24s)
    pivot_signed__in_P/s128   0.03101  n=4096    (8.28s)
    pivot_signed__in_P/s512   0.02881  n=4096   (24.45s)
    pivot_signed__in_P/s1024  0.02966  n=16384 (339.22s)

**5 of 36 units banked.** ADR-001 did exactly what it was written for: the same
600 s kill that destroyed 588 s of work in iteration 13 destroyed nothing here.

**AND IT DOUBLED AS A DETERMINISM AUDIT, unplanned.** Every one of those five
rates is **bit-identical** to the pre-ADR run, from a fresh process and a fresh
interpreter: 0.02466 / 0.02856 / 0.03101 / 0.02881 / 0.02966. The measurement is
reproducible.

**[RUN] FLOOR SENSITIVITY OF THE M2 ARM** -- checked because Inspector S4 says
the published decay exponent is a floor artifact. sign_flip_rate has floor=1e-6,
which discards draws whose gradient is below an ABSOLUTE threshold:

    s=32   floor=1e-6 0.031250   floor=0 0.031250   1.00x
    s=128  floor=1e-6 0.026367   floor=0 0.026367   1.00x
    s=512  floor=1e-6 0.024414   floor=0 0.024414   1.00x

**The floor discards NOTHING from the pivot arm at any s.** S4 is real for sgate,
whose gradients shrink into the threshold, and absent here. A genuine
strengthening of the M2 measurement, checked rather than hoped for.

---

## HEALTH INSPECTOR REPORT — 4 STRIKES. A struck claim LEAVES the verdict.

Clean: the 84x retraction is complete and the paraformer arm now applies gamma
(bench.py:378-382 read directly); 794 tests collectible, 31/31 cited names
resolve, tests/w11 11/11; lake build CEQ exit 0 with **12** theorems all on
[propext, Quot.sound, Classical.choice], no sorryAx; calibration 4/4
bit-identical; 26/26 published probe cells reproduce exactly; Fisher recomputed
as **2.7502788939e-05** against the published 2.8e-05.

**S1 — the windowed "non-vanishing regime" is STRUCK.** Three grounds.
  (i) UNBOUND: its test is GREEN on first appearance and **never once RED**.
  (ii) TRUE BY CONSTRUCTION: with w=8 and j=i-4, c=i-2, out_i depends only on
      [i-16, i]. Holding a fixed 20-token tail and varying s, A[i, i-8:i] is
      identical to ~7 significant figures at s=32/128/512/2048. "Flat across a
      64x context growth" cannot come out any other way.
  (iii) THE DECISIVE CONTROL WAS NEVER RUN AND IT KILLS IT: every windowed number
      is at w=8. Sweeping w at fixed s=512: 0.0996 / 0.0371 / 0.0156 / 0.0078 /
      0.0039 / **0.0000** for w = 8/16/32/64/128/256. Log-log slope IN WINDOW
      WIDTH **-1.159 (R2 0.9933)** -- indistinguishable from the -1.437 decay in
      context. **The property does not survive a bounded field; it survives
      w = 8.** gpt-oss sliding window is 128; Mistral and Gemma are 4096. The
      README sentence "it is the standard bounded field of Mistral, Gemma and
      gpt-oss" is FALSE as written.

**S2 — DeltaNet "non-overlapping 1.39x win" is STRUCK as a floor artifact.**
  sgate row-L1 is pinned at rho=1.5; DeltaNet row-L1 grows ~linearly (4.23 /
  21.53 / 81.62 at s=32/128/512), so the ABSOLUTE floor bites the arms unequally:
      floor=1e-6  sgate 0.026367  deltanet 0.054688   2.07x
      floor=1e-9  sgate 0.045898  deltanet 0.054688   1.19x
      floor=0     sgate 0.050781  deltanet 0.054688   **1.08x, overlapping**
  sgate moves **1.93x**; DeltaNet does not move at all.
  **This does NOT restore any novelty claim** -- DeltaNet still >= sgate at
  floor=0, and SDA 2606.04833 is prior art for the sgate matrix regardless. Only
  the MARGIN is struck. The deletion stands.

**S3 — "the discard floor is not the cause" is STRUCK.** Run at j=1 while every
  published headline uses j=s/4, and at 256 draws whose resolution (1/256 =
  0.00390625) is exactly the value it reports at s=64 and s=128. The instrument
  was incapable of showing a difference precisely where the difference lives. At
  the published geometry the OFF/ON ratio runs 1.04 / 1.11 / 1.93 / 2.25 /
  **3.25x**, and the control own numbers rise monotonically -- the opposite of
  "the same curve".

**S4 — the decay exponent is STRUCK on magnitude; the conclusion survives.**
  floor=1e-6: 0.174805 / 0.088867 / 0.026367 / 0.011719 / 0.003906, slope
  **-1.389**, R2 0.9938. floor=0: 0.181641 / 0.098633 / 0.050781 / 0.026367 /
  **0.012695**, slope **-1.221**, R2 0.9662. **The decay is real**, the exponent
  is inflated, and the s=128 rate is 3.25x higher than published. No document
  states the floor value or that any rate is floor-sensitive.

**INDEPENDENT CONFIRMATION OF tgate.** The Inspector ran its pre-registered kill
himself: **tgate -0.071, tgatex -0.104**, deltanet -0.045, sgate **-1.624**,
softmax 0. Against a kill bar of "slope steeper than -0.3", **tgate passes** --
measured by someone who was not trying to make it pass.

**THE INSPECTOR OWN CHECKLIST -- what is NOT done:**
  * WIN CONDITION unmet: intervention generalization has all arms worse than a
    constant; **ARC-AGI never scored; Turing-style eval never attempted, no file
    exists.**
  * SCALE GATE unmet: nothing above 3.65M params; 300M priced at 912 A100-hours.
  * Three of four original design decisions DELETED (DEQ, successor
    representation, sheaf Laplacian); the fourth survives only in summarizing form.
  * **"The equilibrium is not an equilibrium"** -- the shipped operator is a
    direct triangular solve and says so in its own source.
  * Refcount.lean is the only live provenance candidate and it is **cited in no
    document and gated by no test**.
  * UNVERIFIED: whole-suite pytest tests/ -q has never completed, in this session
    or in the log. ~9% in 15 min on CPU, est. >3 h. **No total pass/fail count
    exists. Do not quote one.**

CHECKLIST: no status changed. M2 is 5/36 units done and remains UNTESTED.


---

## LOOP ITERATION 15 — 2026-08-25 — Inspector clean. Draw-batching added. Bucket 2 ran.

**INSPECTOR PASS (every 5th iteration) — 3/3 CLEAN.**
  1. calibration [RUN] --self-test exit 0, 4/4 bit-identical.
  2. LOCK M2: live CHECKLIST.md hash **efadc390c93f**, matches. Item text has not
     moved since its test began.
  3. re-run published number, selector 15 mod 7 = **1** -> sgate hops=2.
     [RUN] got **0.1640625**, want 0.1640625, **MATCH**.

**ADR-001's "TO REVISIT" CASE HAS BEEN REACHED, and it was reached by
measurement.** Timed s=2048 over **120 draws** (not 4 -- that lesson is on
record): **80.1 ms/draw**, so one 16384-draw unit costs **1312 s** and CANNOT fit
the 600 s platform cap. Such a unit is unrunnable, not merely slow. ADR-001
anticipated this exactly: *"if a single unit ever exceeds the wall-clock cap,
units must be split further (e.g. by draw batch)."*

ACTION (one): split s=2048 into **16 batches of 1024 draws, seeds 0-15**,
aggregated by summing flips and draws.

**THIS IS AN IMPROVEMENT, NOT A COMPROMISE.** Both forms are 16384 independent
draws, so the total is unchanged -- but the batched form spreads **16 seeds**
where the single unit had one. This project has already been burned by a single
seed: a tau sweep on seed 0 produced a false optimum that evaporated to 1-in-5
when replicated. Sizes below 2048 are unchanged and their journalled units stay
valid (s<=512 at 4096 draws ~24 s; s=1024 at 16384 draws 339 s, both under cap).

Unit count rises 36 -> **126** (6 cells x (5 single-unit sizes + 16 batches)).

**BUCKET 2 [RUN]** `python -u scale/m2_units.py --budget 380`:

    replay pivot_signed__in_P/s8: MATCH        <- determinism audit, again
    [6/126]  pivot_signed__in_P/s2048/b0  rate 0.03223  k=33  n=1024   (83s)
    [7/126]  pivot_signed__in_P/s2048/b1  rate 0.03418  k=35  n=1024   (89s)
    [8/126]  pivot_signed__in_P/s2048/b2  rate 0.01855  k=19  n=1024   (94s)
    [9/126]  pivot_signed__in_P/s2048/b3  rate 0.02441  k=25  n=1024   (92s)
    [10/126] pivot_signed__in_P/s2048/b4  rate 0.03223  k=33  n=1024  (106s)
    budget spent (464s); stopping cleanly with 116 units left
    [m2] bucket end: 10/126 done, 116 remaining

**It stopped CLEANLY on its own budget for the first time** -- not killed by the
platform, not losing an in-flight unit. That is the whole point of ADR-001
working in the intended direction rather than the recovery direction.

s=2048 aggregate so far, 5 of 16 batches: k=145, n=5120, **rate 0.02832** --
consistent with the flat ~0.029 at every smaller s. **Still not a result**: the
`c NOT in P` control has not run, and a flat claim beside a flat control is RED.

**COST, STATED HONESTLY:** 116 units remain. The 96 s=2048 batch-units at ~90 s
each are ~2.4 h alone, so M2 needs roughly **20-25 more buckets**. That is a
large share of the loop's remaining iterations, and it is the price of the
pre-registered draw counts. Cutting them to finish sooner would be the silent cap
the rules forbid.

CHECKLIST: no status changed. M2 is 10/126 units done and remains UNTESTED.


---

## LOOP ITERATION 16 — 2026-08-25 — Priority scheduling added. Bucket 3. Fellows re-tasked toward the ship path.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**PRIORITY SCHEDULING — nothing cut, order changed.** At ~90 s per s=2048 batch,
M2 needs 20-25 buckets, which is most of the loop remaining iterations. The
frozen kill reads exactly TWO slopes (c in P, c NOT in P) and neither can be
trusted without both instrument ends (softmax FLOOR, random CEILING). Those four
cells are 84 of 126 units; `dense_signed` and `windowed` are context and CANNOT
change the verdict. So `CELLS` is reordered decisive-first.

**THIS IS NOT THE SILENT CAP THE RULES FORBID, and the distinction is exact: a
cap changes what evidence EXISTS; an order changes only WHEN it arrives.** All
126 units still run. The journal is keyed by unit name, so order affects
scheduling only and never correctness.

Added `--decisive`, which emits the verdict from the four cells the kill actually
reads, REFUSES if any decisive unit is missing, and always prints which
non-decisive cells are outstanding so it can never be mistaken for the full
report.

**BUCKET 3 [RUN]** budget 360 s:
    replay pivot_signed__in_P/s8: MATCH        <- determinism audit, third time
    b5 0.04102 | b6 0.02832 | b7 0.03516 | b8 0.03125 | b9 0.02734 | b10 0.02734
    budget spent (410s); stopping cleanly with 110 units left
    [m2] bucket end: 16/126 done

s=2048 aggregate, 11 of 16 batches: **rate 0.03063** (k=345, n=11264) --
consistent with the flat ~0.029 at every smaller s. **Still not a result**: the
`c NOT in P` control has not run, and a flat claim beside a flat control is RED.

**CONFIRMED FACT, after EIGHT failed attempts across two agents: the whole-suite
`pytest tests/ -q` has never completed.** A dedicated agent tried background
tasks, `tee`, file redirection and a wait loop; every run either hit the 600 s
cap or truncated at 4-17%. `--collect-only` succeeds at **794 collected in
19.80s, exit 0**. **No total pass/fail count exists for this repo. Do not quote
one.** This is now a standing fact, not a pending task.

**FELLOWS RE-TASKED toward the Colab -> HuggingFace path, per the user.** All
three run with nurses that write code for them, scoped to code/docs/Lean so they
do not contend with the M2 buckets, with explicit file ownership so they cannot
collide:
  * **FOREMAN -> research documentation**, invoking `/anthropic-skills:research-readme`
    and `/design:research-synthesis`. Landing his four-strikes root-cause work
    first (the general failure: an ABSOLUTE threshold applied across arms whose
    scale differs and drifts with s is not a discard rule, it is an uncontrolled
    arm-dependent filter).
  * **CHASE -> unit tests and Lean tests.** Priority: bind `Refcount.lean`, which
    the Inspector called "the only live provenance candidate -- cited in no
    document and gated by no test". He had already started
    `tests/chase/test_lean_refcount_binding.py`.
  * **CAMERON -> Colab, HuggingFace, policy, and THE NICHE.** The user's real
    question: which text-generation niche this attention fits. Constrained to
    answer from evidence -- the property softmax structurally cannot have is
    content-conditional sign, and it lives at SHORT range (decays -1.221 in
    context, -1.159 in window width). Told explicitly that "no niche is
    demonstrated yet" is an acceptable answer if that is what the evidence says.
  * **WILSON -> holds.** Dispatched only after the three report, per the user, to
    check scope against what they actually deliver.

CHECKLIST: no status changed. M2 is 16/126 units and remains UNTESTED.


---

## LOOP ITERATION 17 — 2026-08-25 — M2 bucket 4. 20/126.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
Notable: three fellows are editing the repo concurrently (new test files
appearing in tests/chase and tests/foreman, and ceq/attention.py edited) and the
calibration is still bit-identical. G2 has not fired.

ACTION (one): M2 bucket 4, budget 340 s.

    replay pivot_signed__in_P/s8: MATCH        <- determinism audit, fourth time
    b11 0.02930 (95s) | b12 0.03223 (130s) | b13 0.02832 (111s) | b14 0.02930 (60s)
    budget spent (396s); stopping cleanly with 106 units left
    [m2] bucket end: 20/126 done

Per-batch wall time is drifting up (60-130 s against ~90 s in bucket 2) because
three opus fellows plus their nurses share this box. Expected, and the bucket
still stopped on its OWN budget rather than being killed -- which is the property
ADR-001 was built for.

**s=2048 `c IN P`, 15 of 16 batches: k=462, n=15360, rate 0.030078.**
Per-batch spread 0.01855 to 0.04102 across 15 independent seeds -- the batching
is doing its second job, showing the seed-to-seed variation a single-seed unit
would have hidden.

Aggregate so far, `pivot_signed c IN P`:
    s=8 0.02466 | s=32 0.02856 | s=128 0.03101 | s=512 0.02881
    s=1024 0.02966 | s=2048 0.03008 (15/16 batches)

**FLAT ACROSS A 256x CONTEXT GROWTH. STILL NOT A RESULT.** The `c NOT in P`
control has not run a single unit. Under the frozen kill a flat claim beside a
flat control is RED, not a win -- and that control is the very next cell.

CHECKLIST: no status changed. M2 20/126, UNTESTED.


---

## FOREMAN — 2026-08-25 — S2/S3/S4 ARE ONE DEFECT. Structural fix + the two missing controls, pinned.

CALIBRATION [RUN] `python run_calib.py --self-test` -> **exit 0, 4/4 bit-identical**,
run AFTER every edit to `ceq/bench.py` below. G2 did not fire.

### THE ROOT CAUSE, one sentence

`sign_flip_rate` fused its EXCLUSION RULE into its MEASUREMENT LOOP, and that rule
was an ABSOLUTE threshold (`min(|lo|,|hi|) > 1e-6`) applied to gradients whose
scale differs by two orders of magnitude between arms and drifts with `s`, so the
threshold was never a dust filter — it was an uncontrolled, arm-dependent,
`s`-dependent SAMPLE FILTER, and every comparison it gated (S2's arm ratio, S3's
own control, S4's exponent) was reporting the filter as much as the phenomenon.

### THE GENERAL FAILURE, named one level higher

**An exclusion rule that is not invariant under the same group as the statistic it
gates.** The reported quantity is SCALE-FREE — the sign of a number is invariant
under positive rescaling. The rule that decided which draws counted carried UNITS.
Applying a rule with units to a question without them imports each arm's own
magnitude into the answer.

S1 is the SAME failure in a second costume: **an uncontrolled scale parameter.**
There the parameter was the discard threshold, held absolute while the arms'
gradient scale moved; in S1 it is the window width, held at `w = 8` while `s`
moved. In both, the sweep varies the LABELLED variable while a second, UNLABELLED
variable sets the answer. All four strikes are that one sentence.

### THE CAUSE, MEASURED — not inferred from row-L1

[RUN] published geometry `s=32, i=31, j=8, c=16, hops=2, n=1024, seed 0, cpu`:

    arm        median max|grad|     where 1e-6 sits in that distribution
    sgate           0.00425         2.36e-4 of the arm's own scale
    deltanet        0.948           1.06e-6 of the arm's own scale
    softmax         3.20e-12        (zero sign-flipped draws; structural)

**A factor of 223 between the two arms' gradient scales.** One shared absolute cut
lands INSIDE sgate's distribution and five decades BELOW DeltaNet's. Discard
fractions at `floor=1e-6`: **sgate 48.08% of its flipped draws, DeltaNet 0.00%.**
That number is the defect, and no published rate ever carried it.

### THE STRUCTURAL FIX — split MEASUREMENT from EXCLUSION

`ceq/bench.py`. The rule is now a pure function over stored draws, so a floor
sweep costs no re-measurement — which is precisely why no published number ever
carried one.

  * `sign_flip_draws(...) -> list[(lo, hi)]` — the raw measurement. No discard rule
    is applied here and none is available here.
  * `flip_rate(draws, floor=1e-6, rel=0.0)` — the rule.
    `keep if lo*hi < 0 and min(|lo|,|hi|) > max(floor, rel * median(max(|lo|,|hi|)))`
  * `discard_fraction(draws, floor, rel)` — the share of flipped draws thrown away.
  * `sign_flip_rate(...)` — unchanged signature plus `rel`; now a two-line wrapper.
    `rel = 0.0` default reproduces the historical behaviour BIT-FOR-BIT (the draw
    loop and therefore the RNG stream are untouched), which is why G2 stayed green.

**THE PROPOSED CRITERION IS `floor=0, rel=1e-9`.** `rel` is dimensionless and
multiplies the arm's OWN median, so the cut carries the arm's units. Sizing:
float64 eps is 2.2e-16, a two-hop sum over s=32 accumulates order 1e3 operations,
so genuine round-off is order 1e-13 of the row scale; 1e-9 sits four decades above
that. **It is not a knob to be tuned upward** — measured ratio by rule at s=32:
rel=0 **1.077x**, 1e-9 **1.120x**, 1e-6 **1.217x**, 1e-3 **2.391x**, i.e. an
over-aggressive relative rule is WORSE than the absolute floor, because the two
arms' gradient distributions differ in SHAPE as well as scale. Stated as a limit,
not hidden.

### RED-FIRST EVIDENCE — both new files, `--runxfail` forcing the pinned defects to report

`tests/foreman/test_absolute_floor_is_an_arm_filter.py` [RUN]
`pytest -q --runxfail` -> **2 failed, 5 passed**:

    E  AssertionError: the discard rule threw away 48.08% of sgate's flipped
       draws and 0.00% of DeltaNet's. That is an arm-dependent sample filter,
       not a dust guard.
       assert 0.4807692307692308 <= 0.1
    E  AssertionError: the discard rule moved the arm-vs-arm ratio from 1.077x
       to 2.074x. The ratio is reporting the rule.
       assert 0.9259259259259258 <= 0.25

    floor=1e-06 rel=0  sgate 0.026367  deltanet 0.054688  ratio 2.074x
                       (no rule at all: 1.077x)

**Independent replication of S2 to the last digit** — 0.026367 / 0.050781 /
0.054688, 2.07x -> 1.08x, measured here from raw draws rather than taken from the
Inspector.

`tests/foreman/test_window_width_is_the_missing_control.py` [RUN]
`pytest -q --runxfail -s` -> **1 failed, 2 passed**:

    E  AssertionError: the rate falls with window width at slope -1.540. The
       'non-vanishing regime' is a property of w=8, not of a bounded field.
       Mistral and Gemma window at 4096, gpt-oss at 128.
       assert 1.5400732149199117 < 0.3

    sgate, hops=2, 384 draws/cell, cpu, seed 0
      window width swept at s=256:  w=8 0.122396  w=16 0.052083  w=32 0.015625  w=64 0.005208
      context swept at w=8:         s=32 0.130208  s=64 0.104167  s=128 0.125000  s=256 0.122396
      SLOPE IN WINDOW WIDTH  -1.540  (R^2 0.9956)
      SLOPE IN CONTEXT       -0.000  (R^2 0.0000)

**The context slope is EXACTLY FLAT and the window-width slope is -1.540.** S1's
missing control, independently reproduced at a cheaper geometry than the
Inspector's (they measured -1.159 at s=512; both verdicts identical).

Ground (ii) also made structural rather than statistical: with `w=8, j=i-4, c=i-2`
and the same 32-token tail, `A[i, i-8:i]` differs between s=32 and s=128 by
**2.384e-07 against a row scale of 1.201 — 1.985e-07 relative**, which is float32
epsilon from BLAS choosing a different reduction order for a different matrix
SHAPE, not a dependence on the prefix.

SHIPPED STATE [RUN] `pytest -q` both files -> **7 passed, 3 xfailed in 49.59s**.
The three defects are pinned as `xfail(strict=True)`: they MUST keep failing, and
the day one passes the suite goes red and the calibration is revisited
deliberately instead of silently. That is how the missing control stops going
missing.

COLLECTION [RUN] `pytest --collect-only -q tests/` -> **829 tests collected in
39.27s**, exit 0. My two files contribute 10 of them; the rest of the delta from
the previously recorded 794 is not mine to claim.

### AUDIT TABLE — the headline decay curve, README:64, README:133, MODEL_CARD:62

[RUN] sgate, hops=2, n=1024, `j=s/4, c=s/2`, seed 0, cpu. The published row
reproduces **bit-identically**, which is what licenses the rest of the row.

| s | floor=1e-6 (**published, undocumented**) | floor=0 | rel=1e-9 | move | flagged >10%? |
|---|---|---|---|---|---|
| 8 | 0.174805 | 0.181641 | 0.178711 | 1.04x | no |
| 16 | 0.088867 | 0.098633 | 0.098633 | 1.11x | **YES** |
| 32 | 0.026367 | 0.050781 | 0.048828 | 1.93x | **YES** |
| 64 | 0.011719 | 0.026367 | 0.026367 | 2.25x | **YES** |
| 128 | **0.00391** | **0.012695** | 0.012695 | **3.25x** | **YES** |
| slope | **-1.389** (R^2 0.9938) | **-0.958** (R^2 0.9990) | **-0.953** (R^2 0.9989) | | **YES** |

**4 of the 5 published cells and the published slope are floor-sensitive by more
than 10%.** Every one of them appears in README.md and MODEL_CARD.md with no
statement of the floor's value or of its existence.

### A CORRECTION TO THE INSPECTOR'S OWN CORRECTION

S4 reports the floor=0 slope as **-1.221 (R^2 0.9662)**. On the five floor=0 rates
the Inspector themselves published — 0.181641 / 0.098633 / 0.050781 / 0.026367 /
0.012695 at s = 8/16/32/64/128 — OLS in log10 gives **-0.958 (R^2 0.9990)**.
[DERIVED] log10 y = -0.7408 / -1.0060 / -1.2941 / -1.5789 / -1.8964; successive
differences per doubling -0.265 / -0.288 / -0.285 / -0.318, mean -0.289, divided by
log10(2)=0.301 gives -0.96. The two numbers cannot both come from those five
points, and **I could not reconcile them** — most likely the Inspector fitted a
different `s` range and did not say which.

Three consequences, all load-bearing:
  1. **The corrected exponent is about -1, not -1.2.** The rate falls essentially
     as `1/s`, which is the shape "one token's share of a prefix of length s"
     predicts — a cleaner mechanism than the published -1.389 ever suggested.
  2. **The UNFILTERED data fits the power law BETTER** (R^2 0.9990 vs 0.9938). The
     floor was adding curvature, because it bites harder at larger `s` as
     gradients shrink. That is the S4 mechanism confirmed from the residuals and
     not merely from the endpoint.
  3. `rel=1e-9` gives -0.953, indistinguishable from `floor=0`'s -0.958. **The
     relative criterion does not distort the exponent.**

THE DECAY IS REAL AND UNTOUCHED BY ANY OF THIS. Softmax reads exactly 0.0000 under
`floor=1e-6`, `floor=0`, `rel=1e-9` AND `rel=1e-3` alike, with **zero sign-flipped
draws out of 1024** — a structural zero, not a thresholded one. Removing the floor
entirely manufactures no flip from noise. Only the MAGNITUDE was inflated.

### WHERE ELSE THE DEFECT LIVES

**FIXED — three verbatim copies of the defective line, now routed through the one
shared rule** (`bench.flip_rate`), each gaining a `rel` parameter. All three keep
`floor=1e-6` as default, so no published number off them moves:
  * `scale/carpet_probe.py:92` [READ]
  * `scale/sparse_probe.py:89` [READ]
  * `scale/recall_probe.py:99` [READ]

**FOUND, DELIBERATELY NOT TOUCHED — `scale/pivot_probe.py:203`** [READ] is a fourth
verbatim copy. `scale/m2_units.py:34` imports `run_arm` from it, so it is LIVE M2
measurement machinery and out of scope by the standing rule. It is also the one
site already measured BENIGN: DONE.md:2476-2478 records floor=1e-6 vs floor=0 at
1.00x for all of s=32/128/512 on the pivot arm. **It should still be routed through
the shared rule by whoever owns the loop** — benign today is not benign after the
next arm is added.

**SAME DEFECT, DIFFERENT QUANTITY — `scale/dfloor_probe.py:64` and
`scale/dfloor_probe2.py:56`** [READ]: `floor_m` counts DISTINCT readings using an
ABSOLUTE `tol=0.35` on `(r - p).abs().max()`, where `r` is a softmax output — a
convex combination whose magnitude falls as the averaging window grows with `s`.
The distinct-count `m`, the DeltaFloor score, and the reported **slope -0.309 that
fired a pre-registered kill** (DONE.md:1377, 1383) are therefore all conditioned on
an absolute tolerance applied across `s`. v2 fixed a DIFFERENT structural artifact
(constant block COUNT -> constant keep FRACTION) and left `tol=0.35` absolute.
**NOT RE-MEASURED — this is a flagged suspicion, class READ, not a strike.** The
kill fired in the direction of deletion, so a correction here can only make the
deleted selector look better, never worse.

**COMPOUNDING FILTER — `ceq/diagnose.py:87`** [READ]:
`fit = [... for s, r in rate.items() if r > 0.0]`. Zero rates are dropped before
the log-log fit. Dropping them is correct in isolation (log10(0) is undefined and
`_loglog_fit`'s docstring says so). But the CAUSE of a zero rate at large `s` is
the floor, so the exponent is conditioned on the same filter twice — once by
discarding draws, again by discarding the points that discarding produced. The new
window test clamps to `1/(2N)` and prints the clamp instead.

**CHECKED AND CLEARED, so the list is a search and not a fishing trip:**
  * `ceq/attention.py:111` — `floor = finfo(dtype).min / 2` is a dtype-relative
    SENTINEL discriminator for mask conventions, not a magnitude threshold on data.
  * `tests/foreman/test_l1_normalizer_obstruction.py:169` — `atol=1e-12` against
    `rho`, a quantity pinned at 1.5 by construction. Absolute is correct here.
  * `tests/foreman/test_r1_settling.py:295` (`< 1e-9 + 1e-6*max(a,b)`),
    `tests/chase/test_scale_hazards.py:84`, `tests/w8/test_w8_real_model.py:172,195`
    (`< 1e-3 * stock`) — already scale-relative. These are the house style the
    probes should have followed.
  * `tests/w8/test_w8_real_model.py:107` — `(a1-a0).abs().max() > 1e-4` on attention
    weights whose scale goes as `1/s`. Absolute, and it would drift if that test
    ever swept `s`. It does not today. **Flagged, not struck.**

### WHAT I COULD NOT VERIFY

  * **The Inspector's floor=0 slope -1.221.** My fit on their own five rates gives
    -0.958. Unreconciled; see above. Someone should state the `s` range.
  * **The windowed table itself** (README:462-465, s up to 2048 at 2048 draws) was
    NOT re-measured for floor sensitivity. Its s=2048 cell is priced at ~80 ms/draw
    on record, i.e. ~2.7 min for that cell alone, which is outside the CPU budget
    this session runs under. **Its floor sensitivity is UNKNOWN, not clean.**
  * **The 26 remaining published probe cells** in README/MODEL_CARD (lines 377-383,
    419-426, 462-465, and the nine-setting knob grid at 152-157) were not swept for
    floor sensitivity. Only the headline decay curve was. Given 4 of 5 headline
    cells moved by >10%, the prior for those is not good.
  * **Whole-suite pass/fail.** Not attempted. `--collect-only` is the only whole-
    suite number here: 829 collected. **No total pass/fail count exists.**
  * `scale/dfloor_probe*.py`'s `tol=0.35` sensitivity — reasoned from source, not
    measured.

### DOCS — NOT WRITTEN BY ME, DELIBERATELY

The audit table above is a **handoff to CAMERON**, who owns README.md and
MODEL_CARD.md. Nothing in either file was edited by FOREMAN. The four flagged cells
and the slope need the floor value stated where they live; the correct floor=0
exponent to publish is **-0.958**, not -1.221.

---

## CHASE ROUND 4 — 2026-08-25 — Lean provenance bound; the tier numbers withdrawn and re-measured; Colab->HF chain handed to CAMERON

Reassigned mid-run to unit tests and Lean tests. The Colab->HF findings below are
the handover; `ceq/hf/*` was not edited by me.

### PART A — COLAB -> HF CHAIN, HANDOVER TO CAMERON

**[RUN] The relative import is safe on the Hub path and broken on every other
path.** Copied ONLY `configuration_ceq.py` and `modeling_ceq.py` into a directory
outside the repo, scrubbed the repo root from `sys.path`, cwd not the repo:

    naive  spec_from_file_location("modeling_ceq", ...)
      -> ImportError: attempted relative import with no known parent package
         at ceq/hf/modeling_ceq.py:84  `from .configuration_ceq import CEQConfig`

    Hub-style dotted load -> OK; CEQForCausalLM loads,
                             mod.CEQConfig is cfg.CEQConfig  True

The Hub path works because `transformers/dynamic_module_utils.py` (5.3.0) does
two things a naive import does not: `create_dynamic_module` touches an
`__init__.py` in the cache directory (lines 101-120) and `get_class_in_module`
builds a **dotted** module name from the cache path before
`spec_from_file_location` (lines 266-311). So `trust_remote_code` is fine and a
reader who downloads the two files and runs `python -c "import modeling_ceq"` is
not. **Not a defect to fix** -- every custom-code repo on the Hub is like this --
but it is the first thing a stranger tries.

**[RUN] Every module-level import resolves on `torch` + `transformers` alone.**
10 of 10 statements imported clean in isolated subprocesses: `math`, `torch`,
`torch.nn.functional`, `torch.nn`, `transformers.PreTrainedModel`,
`transformers.generation.GenerationMixin`,
`transformers.modeling_outputs.{BaseModelOutput,CausalLMOutputWithPast}`,
`transformers.PretrainedConfig`, plus two `__future__` lines. Line 84 is the only
relative one.

**[RUN] The `trust_remote_code` round trip is BITWISE.** CEQConfig(vocab 64,
hidden 32, 2 layers, 2 heads, max_pos 32), forward on `arange(8)`, save, reload
through `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`:

    os.listdir  ['config.json', 'configuration_ceq.py', 'generation_config.json',
                 'model.safetensors', 'modeling_ceq.py']
    auto_map    AutoConfig -> configuration_ceq.CEQConfig
                AutoModelForCausalLM -> modeling_ceq.CEQForCausalLM
    torch.equal(logits_before, logits_after)   True
    (a-b).abs().max()                          0.0
    named parameters torch.equal               25/25 True
    any parameter .is_meta                     0/25

The meta-device tying defect found last round stays fixed on this path.

**[RUN] `python -m ceq.hf.smoke` with `CUDA_VISIBLE_DEVICES=""`: exit 0, 7/7.**
Still green on python 3.11.9 / torch 2.5.1+cu121 / transformers 5.3.0, no GPU,
no Triton imported by the package itself (triton arrives via transformers).

**[RUN] VERSION PINNING IS THE THING THAT BREAKS A STRANGER.** Searched the tree
for `requirements*.txt`, `setup.py`, `pyproject.toml`, `environment.yml`:
**none exist**. The only version constraint in the entire project is one notebook
line:

    colab/train_ceq.ipynb:82
    !pip -q install -U 'transformers>=5.0' datasets huggingface_hub accelerate

Lower bound only, **no upper bound, and no `torch` pin at all**, with `-U` so
Colab resolves to whatever is newest on the day the notebook runs. What is
exposed to that: `ceq/hf/modeling_ceq.py:431-436,519` sets six PRIVATE
`PreTrainedModel` class attributes -- `_supports_sdpa`, `_supports_flash_attn`,
`_supports_flex_attn`, `_can_compile_fullgraph`, `_no_split_modules`,
`_tied_weights_keys`. [RUN] all 14 probed names exist in 5.3.0. None is covered
by a `>=5.0` floor. `_tied_weights_keys` is written as a **dict** here and was a
**list** in transformers 4.x -- that is the shape that changes silently.
Suggested fix is one line: `'transformers>=5.0,<6'` plus a torch pin.

`AttentionInterface` / `AttentionMaskInterface` both import fine in 5.3.0 and are
**not touched by the shipped modeling file at all** -- only by `ceq/attention.py`
and `ceq/hybrid.py`, which the Hub never copies. The Hub artifact is not exposed
to that API; MODEL_CARD's `hybrid.register()` Usage block is.

**Also for CAMERON:** MODEL_CARD.md's Reproduce block still opens with
`python -m pytest tests/ -q`. That suite has never completed. A stranger's first
command is a multi-hour hang.

### PART B — THE LEAN PROVENANCE, BOUND

**RED [RUN], the measurement that justifies the file.** Deleted
`free_face_floor_unchanged` and `shared_plaque_floor_drops` -- the corollary the
whole object exists for and the converse that stops it being vacuous -- from
`lean/CEQ/Refcount.lean`, taking it from **10 theorems to 8**. `lake build CEQ`
returned **exit 0 in 32.9 s** and nothing in `tests/w3b` moved. Restored; rebuild
exit 0, 10 theorems. A `sorry` grep cannot see a deletion and neither can a
build, because nothing in the library imports these theorems.

**CORRECTION TO THE INSPECTOR'S VERDICT [READ].** "Gated by no test" is half
right. `tests/w3b/test_w3b_lean_nilpotent.py` globs `CEQ/*.lean` for `sorry` and
runs `lake build CEQ`, and `CEQ.lean` imports `CEQ.Refcount`, so **compilation
and sorry-freedom were already gated**. What was ungated: deletion, rename, a
changed axiom basis, the arithmetic content, the correspondence to shipped code,
and the citation.

**BOUND: `tests/chase/test_lean_refcount_binding.py` — 10 tests, [RUN] 46.65 s,
CPU.** Two `lean` subprocesses at ~23 s each; the cost is olean loading, so every
`#print axioms` goes in one file rather than one per module.

  * **[RUN] all 27 theorems** across `Contraction`/`Nilpotent`/`Occupancy`/
    `OrbitBound`/`Refcount` rest on a subset of `[propext, Quot.sound,
    Classical.choice]`; no `sorryAx` anywhere. The theorem list is DISCOVERED
    from the sources, so a proof added tomorrow is covered the day it lands.
  * **BOTH halves of the detector calibrated in the same process.** A literal
    `theorem ... := by sorry` must report `sorryAx`, and a real proof sitting
    under the comment "No `sorry` anywhere below this line" must report nothing.
    That second half is the exact false positive a previous grep-based detector
    produced on CEQ.lean's own sentence.
  * **A separate hand-written `LOAD_BEARING` list of 14 names** is what catches a
    deletion; discovery cannot, because it enumerates what is there.
  * `import CEQ.Refcount` must be present in `CEQ.lean` -- without it the w3b
    build gate silently stops covering the file while still reporting exit 0.
  * The `caustic` DOI in Refcount.lean must equal the one in OrbitBound.lean.
  * **[RUN] the identity re-derived in Python**, exhaustively over all 3^5 = 243
    maps from 5 prefixes to 3 plaques and 200 random larger ones:
    `n - m = sum over plaques of (refcount - 1)`. Plus the corollary and its
    converse over 400 random caches: evicting refcount 1 moves the floor by
    exactly 0, evicting anything else strictly lowers it, both branches exercised.
  * **[RUN] the truncated-subtraction hazard the header names is real**:
    at `n = m = k = 3` the subtractive form reads left 0, right 2. Reproduced,
    with the control that no map from 3 prefixes onto 3 distinct plaques has a
    fibre of size 3 -- which is why the theorem is stated over `survivors`.
  * `HopCache.evict` raises `RuntimeError("... free face ...")` for a position in
    an enclosing scope and succeeds for one in the current scope.

**RECORDED DIVERGENCE, stated rather than papered over.**
`CEQ.Refcount.IsFreeFace` is `refcount f p = 1`, a fibre cardinality.
`ceq/hopcache.py::HopCache.evict` admits on **scope depth**. [RUN]
`grep -rn refcount --include=*.py` over the tree hits documentation prose only --
**no refcount is maintained anywhere in the shipped Python**. So these theorems
certify the CRITERION, not this implementation of it. The test asserts that
absence, so the day a real refcount ships the test fails and forces the rewrite.

### PART C — TWO DOCSTRING NUMBERS WITHDRAWN, TWO BOUND

**BOUND: `tests/chase/test_module_prose_is_bound.py` — 7 tests, [RUN] 7.5 s, CPU.**

**RED [RUN].** `grep -rn "0.376137914|0.787332|0.808386"` over the whole working
tree returns exactly two hits: `ceq/attention.py:17-18`, and one
`{"t": "finding", "agent": "Foreman", ...}` line in `house-events-round1.jsonl`.
That JSONL is a transcript of what an agent SAID -- no seed, no shape, no script,
nothing rerunnable. **Zero tests mentioned any of the three.** The numbers
licensing the entire architecture ("both baselines sit in tier 2; the bar is tier
3") rested on a quoted sentence for three rounds.

**WITHDRAWN and replaced** with figures the test produces [RUN, seed 0, S=8 D=16
hops=2, float64, both arms fed identical logits so the operator is the only free
variable]:

    tier 1  pair ratio  0.200257204381 -> 0.200257204381   |delta| 1.110e-16
            control: the same perturbation moved the row by 5.349e-01
    tier 2  influence ratio 0.846220 -> 0.425642  (49.70% move)
            0 of 200 draws reach a negative influence entry, min 0.000000e+00
    tier 3  107 of 200 draws reach one, global min -1.377561e-01

**NEW FINDING, and it corrects the header's own logic. THE RATIO IS NOT THE
DISCRIMINATOR.** On the identical draw the SIGNED arm's influence ratio moves
**90.30%** against the non-negative arm's **49.70%**. Both move. So the criterion
that separates tier 1 from tier 2 does **not** separate tier 2 from tier 3 --
only the sign of the minimum influence entry does, and the 0/200 control is what
gives the 107/200 its meaning. The old header presented the ratio move as the
tier-2 marker, which reads as though it were doing discriminating work.
`ceq/attention.py`'s header now says this and names the test; the test asserts the
three withdrawn figures have not come back.

**`ceq/lm.py`'s globals are bound in BOTH directions, not corrected.** [RUN]
`(lm.RHO, lm.SGATE_LAM, lm.HOPS) == (0.9, 1.0, 3)` against
`PARITY_POINT == (1.5, 0.10, 2)`. How far apart, as numbers: max
`|A_parity - A_committed| = 1.2273`, **2.9701** in relative Frobenius norm, and
the difference is structural rather than a magnitude -- committed `lam = 1.0`
makes every row sum exactly zero (max `|row sum|` **8.327e-17**, it annihilates
the constant vector) while parity `lam = 0.10` leaves `rho(1-lam)/(1+lam) =`
**1.227273** of positive mass in every row. The grep bind requires the sentence
"THE MODULE GLOBALS BELOW ARE NOT THE PARITY POINT" to stay in the module.

**NOT CORRECTED, deliberately.** Every number already recorded in `ceq/lm.py` was
measured at those globals, and `ceq/bench.py` -- the source the live calibration
probe reads through `run_calib.py` -- is another agent's file this round. Moving a
module global to make a docstring true would invalidate recorded numbers to fix
prose. The test fails if either side moves without the other.

**[RUN] `python run_calib.py --self-test`: exit 0, 4/4 bit-identical** (signed
hops=3 0.0468750000, sgate hops=1 0.0234375000, sgate hops=2 0.1640625000,
softmax hops=3 0.0000000000), and the gate's own wrong-target self-test still
returns exit 1 as required. No G2. Also re-ran `tests/w11` + `tests/w3b`:
**[RUN] 17 passed in 43.8 s**, so the `ceq/attention.py` docstring edit broke no
existing claim test.

### OPEN — what I could not verify

1. **`tests/chase/test_scale_axes.py` stays RED and should.** [RUN] 5 failed, 2
   passed, 2 xfailed on CPU. `tests/chase/scale_axes.jsonl` holds **3 rows**
   against `EXPECTED = dict(seq=4, depth=4, heads=4, rho=8)` = **20**. The three
   recorded rows carry `"seconds": 299.9 / 786.3 / 1072.5` on CUDA, so the 17
   missing points are **hours of GPU**, unreachable under the CPU rule.
   I did not loosen the assertion: `assert len(got) == EXPECTED[axis]` failing
   rather than skipping is the correct design and the message already names the
   command that generates the data.
2. **`CEQForCausalLM` has still never been trained to 1.0334.** The parity run was
   `lm.TinyLM`. Architectures match and the operator is bitwise equal (smoke check
   2/7), but the end-to-end run in the shipped class has not happened. GPU work.
3. **The Refcount theorems certify a criterion the shipped cache does not
   implement.** Either `HopCache` grows a real refcount or the divergence stays
   documented. I did the second; the first is a design decision, not a test.
4. **13 of the 27 Lean theorems are axiom-gated but not deletion-gated.** Only the
   14 in `LOAD_BEARING` are. The remaining 13 are helper lemmas cited nowhere;
   adding them to the list would make every internal refactor a test failure.
5. **MODEL_CARD.md — BOUNDARY CROSSING, DISCLOSED.** I edited its "Verified core"
   section (theorem count `Seven` -> `**27 theorems**`, a `CEQ.Refcount`
   provenance paragraph, and the "what that does and does not certify" caveat)
   **before the reassignment message landed**. CAMERON owns the file now. The
   edit survived a concurrent write by CAMERON. `test_lean_refcount_binding.py`
   asserts three things about that file -- `CEQ.Refcount` present, the DOI
   present, and `**27 theorems**` equal to the live count -- so CAMERON should
   know before rewriting the section. Say the word and I will revert it.
6. **Version pinning and the Reproduce block are CAMERON's** per Part A; both are
   one-line fixes and neither is done.

CHECKLIST: no status changed. Nothing in `scale/`, `results/`, `run_calib.py`,
`ceq/bench.py`, `STATE.md` or `CHECKLIST.md` was touched.


---

## CAMERON — 2026-08-25 — 4 strikes removed from the docs; Colab priced; niche named.

### 1. THE FOUR STRUCK CLAIMS ARE OUT OF README.md AND MODEL_CARD.md

Applied as 26 exact-match replacements, each asserted to match exactly once, atomic per file.
Every removal states the withdrawal in the document rather than deleting the sentence quietly.

  * **S1 — windowed "non-vanishing regime": WITHDRAWN.** README:511-541. Three grounds now
    printed in the doc: true by construction (`A[i, i-8:i]` identical to ~7 s.f. across
    s=32/128/512/2048 at w=8, j=i-4, c=i-2); its test GREEN on first appearance and never RED;
    and the never-run control — window-width sweep at fixed s=512 reads
    0.0996/0.0371/0.0156/0.0078/0.0039/0.0000 for w=8..256, slope **-1.159 (R2 0.9933) in
    window width**. The FALSE sentence "the standard bounded field of Mistral, Gemma and
    gpt-oss" is DELETED and named as false; those windows are 4096/4096/128, and at w=128 the
    sweep already reads 0.0039. The GREEN test is now cited as "unbound ... evidence of
    nothing" rather than as support.
  * **S2 — DeltaNet's "non-overlapping 1.39x win": WITHDRAWN as a floor artifact.**
    README:56-62, 456-474, and MODEL_CARD:224-240. The floor table now ships in both docs:
    at s=32, floor=1e-6 gives 2.07x, floor=1e-9 gives 1.19x, floor=0 gives **1.08x,
    overlapping**; sgate moves 1.93x, DeltaNet does not move. The s=128 16,384-draw
    separations are withdrawn, NOT restated smaller. Both docs now say in terms: **this
    restores nothing** -- DeltaNet still >= sgate at floor=0 and SDA 2606.04833 Eq. 1-2 IS the
    sgate matrix, so the novelty deletion stands on prior art, not on a margin.
  * **S3 — "the discard floor is not the cause": WITHDRAWN.** README:150-160. The doc now
    states why the old control was blind (j=1 against headlines at j=s/4; 256 draws whose
    resolution 1/256 = 0.00390625 *equals* the value reported at s=64 and s=128) and carries
    the real ratio, **1.04 / 1.11 / 1.93 / 2.25 / 3.25x** over s=8..128.
  * **S4 — decay exponent -1.389: WITHDRAWN.** README:70-84, 143-149, 424-426, 456,
    MODEL_CARD:64-78. Every rate table in both documents now carries its `floor` value, which
    **no document stated before this round**. The floor=0 curve
    (0.181641/0.098633/0.050781/0.026367/0.012695) ships beside the floor=1e-6 one.

  **[DERIVED] A DISCREPANCY IN THE STRIKE REPORT ITSELF, and it is published rather than
  papered over.** The Inspector's replacement exponent **-1.221 (R2 0.9662) does not reproduce
  from the five floor=0 rates it quotes**. Least squares on exactly those five points gives
  **-0.958 (R2 0.9990)** -- each doubling of s roughly halves the rate (ratios 1.84 / 1.94 /
  1.93 / 2.08), which is s^-0.96, not s^-1.22. The same fitter reproduces the Inspector's
  window-width slope to 3 decimals (-1.160 vs -1.159, R2 0.9933 both) and the old -1.389 (R2
  0.9938) exactly, so the fitter is not the problem. **Consequence: NO replacement exponent is
  published in either document.** Both state the decay is real (0.181641 -> 0.012695 is 14.3x
  over a 16x context growth), name both candidate exponents, and say they disagree.
  Downstream: the floor=0 extrapolation reaches ~1.8e-3 at s=1024 and ~9.3e-4 at s=2048, so
  the retired line "below the resolution of 10,000 draws" is **not** true at floor=0 and has
  been removed rather than re-derived.

  **[RUN] The struck numbers still ship inside the package and I cannot fix it.**
  `ceq/hf/modeling_ceq.py::COSTS["content_conditional_sign_decay"]` carries
  `"slope": -1.389, "r2": 0.9938` and the s=8..128 floor=1e-6 rates with no floor stated, and
  `COSTS["capability"]["fisher_one_sided_p"]` is 2.8e-05 against the recomputed
  2.7502788939e-05. `ceq/diagnose.py` hard-codes `published_slope=-1.389` and
  `tests/cameron/test_diagnose_package.py:32` pins it with `SLOPE_TOL`. Both files are outside
  my write scope. **A struck number inside the shipped package is the worst available
  outcome**, so both documents now say so at the point of use (README:757-761 flags COSTS by
  name as "must be corrected before this package is pushed to the Hub"). Owner: CHASE for
  `ceq/hf/*`. Note the real fix is not a constant swap: the tool MEASURES at floor=1e-6, so
  changing the pinned constant alone would turn the test red.

### 2. CORRECTIONS FOUND WHILE REMOVING THE STRIKES

  * **[RUN] MODEL_CARD.md:291 cited a test in the wrong file.**
    `tests/w8/test_w8_real_model.py::test_both_interfaces_are_registered` actually lives in
    `tests/w6/test_w6_attention.py`. It passed the w11 gate because w11 matches on the BARE
    name, so a wrong path is invisible to it. Fixed. (Gap worth knowing: w11 does not check
    the file half of a node id, and does not read MODEL_CARD.md at all.)
  * **[RUN] Fisher p** is now the exact 2.7502788939e-05 in both docs, with the 2.8e-05
    rounding named as a rounding.
  * **[READ] Lean count reconciled to CHASE's audit.** I first wrote a 12-vs-28 "unreconciled"
    note; MODEL_CARD had been updated in parallel to **27 theorems across five modules**, with
    `CEQ.Refcount` now cited and gated by `tests/chase/test_lean_refcount_binding.py`. README
    now matches that, and the Inspector's "Refcount is cited in no document and gated by no
    test" is recorded as **no longer true**.
  * **[RUN] 809 tests collect** in 19.90 s. Both Reproduce blocks now carry
    `pytest --collect-only` as the command that finishes and annotate `pytest tests/ -q` as
    ">3 h, has NEVER completed". Both docs state that **no total pass/fail count exists**.

  **[RUN] GATE: `tests/w11` 11/11 PASSED in 88.22 s** after every edit above. Every test name
  cited in README.md resolves against pytest's own collection.

### 3. [RUN] THE COLAB VERDICT — "912 A100-HOURS" IS UNSUPPORTED, AND THE GATE IS ~130-257

  **"912" has no derivation anywhere in this repository.** It occurs in exactly three places
  (DONE.md:2549, STATE.md:151, README.md pre-edit), each a one-line summary, and no script,
  test or notebook computes it. The repo's own sizing module, whose inputs are explicit,
  disagrees by 3.5-7x.

  Computed from `ceq/sizing.py` (`TOKENS_PER_PARAM=20` x 302,088,192 non-embedding params =
  6.042e9 tokens; A100 bf16 peak 312e12; MFU 0.40 for the CONTROL only; 4 forward-equivalent
  passes with checkpointing), with the operator's cost taken as the control's times the
  **measured** step ratio (3.13x at seq 1024, 5.49x at seq 2048,
  `tests/chase/test_scale_sizing.py:497`) exactly as `gpu_hours`'s own docstring instructs:

      seq 1024   softmax  41.4 A100-h   sgate  129.7 A100-h   5.4 sessions   ~1945 CU  ~$194
      seq 2048   softmax  46.8 A100-h   sgate  257.2 A100-h  10.7 sessions   ~3858 CU  ~$385

  (CU/$ at ~15 CU per A100-hour and 100 CU per $9.99 -- Colab's own pricing page needs
  sign-in, so those two rates are CITED from secondary summaries, not fetched from Google.)

  **The gate is not blocked by money. It is blocked by 15 lines of missing code.**
  `ceq/hf/train.py::train()` trains `steps` from scratch and `save_pretrained`s at the end.
  There is **no resume**: no optimizer state saved, no step counter, no load path. A run
  needing 5.4-10.7 sessions against Colab's 24-hour cap cannot survive the gap between them.
  No Colab tier guarantees an A100 either.

  **[RUN] AND THERE IS A FREE 7x STEP NOBODY HAS TAKEN.** The shipped notebook's own default
  shape (hidden 512, 8 layers, 8 heads, seq 512, batch 8, byte vocab) is **25,707,520
  parameters** -- 7.0x above the 3.65M ceiling that is the most this project has ever trained.
  At a full Chinchilla budget (5.036e8 tokens) it costs **2.3-3.7 T4-hours** for the operator
  and 1.3-2.0 for the control (65 TFLOP/s T4, MFU 0.25-0.40 assumed and labelled), and
  `sizing.fits` says it **fits**: 2.45 GiB activations against a free T4's 14.5 GiB budget at
  batch 8, no gradient checkpointing. One free 12-hour session. As shipped the notebook runs
  2000 x 8 x 512 = 8.19e6 tokens = **1.6% of that budget** -- a smoke run, not a training run.

  **[RUN] IMPORT AUDIT: CLEAN.** By AST, `ceq/hf/modeling_ceq.py` imports only stdlib, torch,
  transformers and one `level=1` relative `.configuration_ceq`; `configuration_ceq.py` imports
  only `transformers.PretrainedConfig`. Every `ceq.` string in both files is inside a comment.
  The two files the Hub copies resolve with torch + transformers alone. `train.py` needs
  `datasets` and `huggingface_hub`, and is NOT copied to the Hub; the notebook pip-installs
  both. No dangling paths in the notebook. `REPO_URL` and `REPO_ID` ship empty, so it cannot
  push anywhere by accident.

  **[READ] One honesty item not yet fixed in the docs:** 36.31 / 33.53 / 20.25 GiB are
  PROJECTIONS from `ceq/sizing.py` calibrated on a 4060, not A100/L4 measurements, and both
  docs read as if measured. Left as-is this round; flagged.

### 4. [CITED] HUGGING FACE POLICY CHECKLIST (fetched 2026-08-25)

  Done this round in MODEL_CARD.md:
  * metadata block now carries `license: mit`, `library_name: transformers` (HF: repos created
    after Aug 2024 must set this explicitly), **`pipeline_tag: text-generation`** -- which is
    also the user's own question answered in metadata -- `language: [en]`,
    `datasets: [roneneldan/TinyStories]`, tags incl. `negative-results`.
  * `## Model Details`, `## Uses` with `### Direct use` and `### Out-of-scope use`,
    `## Training Details`, `## Environmental Impact`, `## Citation`, contact line. Those are
    HF's own annotated-template section names.
  * **`trust_remote_code` disclosed twice, plainly**, in Model Details and in Limits, with the
    two things HF's own docs tell an author to say: it executes code from the repo, and the
    reader should pin `revision=<commit hash>` so a later push cannot change what runs.
    (transformers docs: "Take extra precaution when loading a custom model... As an extra
    layer of security, load a custom model from a specific revision".)

  STILL MISSING, for whoever pushes:
  * On the Hub the model card must be named **`README.md`** in the model repo. MODEL_CARD.md
    is the source; the push step must rename it, and this repo's README.md is a different,
    much longer document. Nothing automates that today.
  * `model-index` structured results are absent. Optional, and putting a 0.0000 in a
    leaderboard widget is a real decision, not an oversight -- left to the user.
  * No `LICENSE` file in the repo, only `license: mit` in metadata and "MIT." in prose.
  * HF's malware/pickle scanning covers uploaded files; the docs carry **no** statement that
    arbitrary custom `.py` modeling code is scanned for behaviour. That is the user's risk to
    disclose, and it is now disclosed.
  * NOTHING WAS UPLOADED. No authentication, no push, no repo created.

### 5. THE NICHE ANSWER — named, addressed, and NOT demonstrated

  **The strong framing is false and this repository already measured it false.**
  `tests/cameron/test_negation_is_the_axis.py::test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`
  shows **one GELU between two softmax layers restores the sign flip**. The theorem is about
  non-negative operators with LINEAR value paths; a real transformer has an MLP. So the
  surviving claim is depth-and-parameter EFFICIENCY -- one layer versus two-plus-a-nonlinearity
  -- not capability. That is now stated in both documents before any niche argument.

  **Three constraints, from measurements:** short discriminating span (rate 0.181641 at s=8,
  0.050781 at s=32 at floor=0; decays in window width at -1.159 too); scarce parameters
  (efficiency claims are invisible when parameters are plentiful); scored on the model's own
  next-token probabilities (it is a text generator, and the property lives in the attention
  block, not a task head).

  **The one candidate that satisfies all three: BLiMP minimal-pair NPI licensing under
  negation, inside the BabyLM strict-small (10M-word) regime.** Published accuracies, BLiMP
  TACL 2020 (5-gram / LSTM / TXL / GPT-2 / human): `sentential_negation_npi_scope`
  **45 / 23 / 53 / 95 / 81**; `only_npi_scope` 30 / 36 / 45 / 85 / 72. The
  `..._licensor_present` paradigm is at ceiling for everything (93/100/99/89/93) and separates
  nothing -- the `_scope` paradigms are the ones where models fail, which is exactly the case
  where a third token must decide whether another token is licensed. TinyStories
  (arXiv:2305.07759) publishes GPT-4-graded scores at 1M / 2.5M / 8.3M / 28M / 33M parameters,
  which brackets both 3.65M and the notebook's 25.7M, so a second published ladder exists.
  **A pre-registered KILL is written into README.md.**

  **Ruled OUT, with reasons:** general LM (lost COGS, gap widened with budget); long context
  (decays in context and in width); ARC-AGI (`tests/cameron/test_arc_reality.py` derives a ~5%
  noise floor from the scoring rule); SCAN (both arms 0.0000); negation-scope TAGGING --
  CD-SCO, BioScope, SFU -- because every published baseline there is SVM/CRF or BERT-base at
  110M, so a 3.65M from-scratch generator has no comparison frame; instruction-override /
  prompt-injection, because **no published benchmark in that family reports a small
  from-scratch model at all** (HackAPrompt, PINT and the rest evaluate production LLMs).

### 6. WHAT I COULD NOT VERIFY

  * The Inspector's **-1.221** exponent (see 1). Refuted by my own fit of its own five rates;
    not silently replaced.
  * **27 vs 12 theorems** -- I take CHASE's 27, which is gated by a test; I did not run
    `lake build` myself.
  * The floor=0 rates, the 1.04..3.25x ratio row, the w-sweep, the 4.23/21.53/81.62 row-L1
    growth and the 7-significant-figure window identity are all **CITED from the strike
    report**, not re-run here. Re-running them needs `ceq/bench.py`, which is FOREMAN's.
  * DeltaNet at floor=0 for any s other than 32; the ParaFormer and Cog/SignGT columns have
    **never** been measured at floor=0. Both docs now say so.
  * Colab's compute-unit rate and per-tier GPU availability: sign-in required, so CITED from
    secondary summaries.
  * No BLiMP number was produced here. The niche is a hypothesis with an address.


---

## FOREMAN — 2026-08-25 — RESEARCH.md written. Six citations resolved; two record corrections.

Reassigned mid-run to documentation. Skills invoked as instructed:
`/anthropic-skills:research-readme` (structure and standards taken from it) and
`/design:research-synthesis`.

**`/design:research-synthesis` DOES NOT FIT THIS MATERIAL AND WAS NOT FORCED TO.**
It is a USER-research skill: its inputs are interview transcripts, survey CSVs,
usability notes, NPS responses, and its output template is participant quotes,
user segments and prevalence counts ("5 of 8 participants"). There are no
participants here. Three of its disciplines transfer and were used: **separate
observations from interpretations**, **quantify instead of saying "most"**, and
carry an explicit **methodology-limitations** section. The rest was left alone
rather than dressed up — inventing "user segments" for an attention operator
would be the same class of error as the instruments this project keeps catching.

DOCUMENT: **`RESEARCH.md`**, 299 lines. README.md and MODEL_CARD.md were NOT
edited — both are CAMERON's.

### WHAT IT CARRIES

The honest position, not softened: nothing here is novel attention; the capability
number is a LOSS (COGS-gen softmax 0.0293 = 15/512 vs sgate 0.0000 = 0/512 at
3,652,096 matched parameters, and the deficit is already present in-distribution
0.9258 vs 0.7734, so it is not a generalization story); ARC-AGI never scored;
Turing-style eval never attempted; nothing above 3.65M against a 300M gate; four
Inspector strikes, and a struck claim leaves the verdict.

Every claim carries its evidence class. Numbers re-derived this session rather
than copied:

  * [RUN] one-sided Fisher, recomputed from the two counts:
    **2.7502788939116803e-05** against the published 2.7502788939e-05. MATCH.
  * [RUN] `zero_success_upper_bound(512) = 0.005834` — sgate's "0.0000" is
    consistent with true skill up to 0.58%, stated so the zero is not oversold.
  * [RUN] `resolution_floor(512) = 0.001953`,
    `min_successes_for_separation(512) = 5`. Softmax's 15 clears the bar; 0 does not.
  * [RUN] `pytest --collect-only -q tests/` -> **829 in 39.27 s**, exit 0.

### [CITED] SIX arXiv IDs RESOLVED — the novelty verdict now has provenance

Resolved against arxiv.org this session; **6 of 6 real, titles match the claims**:

    2206.08898  SimA: Simple Softmax-free Attention for Vision Transformers   17 Jun 2022
    2606.04833  Signed Dual Attention: Capturing Signed Dependencies in
                Time Series Forecasting   Courvoisier & Cazenave   3 Jun 2026
    2406.06484  Parallelizing Linear Transformers with the Delta Rule over
                Sequence Length (DeltaNet)   Yang et al.   10 Jun 2024
    2512.14619  ParaFormer (PageRank-enhanced attention module)   Dec 2025
    2411.07176  More Expressive Attention with Negative Weights (Cog Attention)
                Lv et al.   11 Nov 2024
    2310.11025  SignGT: Signed Attention-based Graph Transformer   17 Oct 2023
    2307.08621  Retentive Network: A Successor to Transformer   17 Jul 2023

**WHAT THIS DOES NOT ESTABLISH, stated in the document itself.** Abstracts confirm
the identifiers and titles. They do NOT confirm the equation-level claims — that
SDA's matrix IS sgate's, that DeltaNet's WY inverse takes the stated form, that Cog
reduces to `sgn(p)*softmax(|p|)`. Those need full texts, which were not retrieved.
Carried as [READ] from this repo's record, not upgraded to [CITED].
Scholar Sidekick was tried first and returned "not subscribed".

### TWO CORRECTIONS TO THE RECORD, both found while writing

**1. "12 theorems" is STALE. It is 27.** [READ] `rg "^theorem " lean/CEQ/*.lean`
-> 27 declarations, 0 `lemma`, no `sorry`: Nilpotent 4, Occupancy 3, Refcount 10,
Contraction 5, OrbitBound 5. CHASE's round-4 entry independently confirms 27 by
`#print axioms` on all of them — `[propext, Quot.sound, Classical.choice]`, no
`sorryAx`. The discrepancy is resolved in favour of 27; `lake build` was NOT re-run
here. Toolchain [READ]: `leanprover/lean4:v4.7.0`, `mathlib4 @ v4.7.0`.

**2. "Refcount.lean is gated by no test" is NO LONGER TRUE** and RESEARCH.md says so
rather than repeating the Inspector. Per CHASE [READ]: `tests/w3b` already gated
compilation and sorry-freedom via `CEQ.lean`'s import; `tests/chase/test_lean_refcount_binding.py`
now binds deletion, rename, axiom basis, arithmetic content, code correspondence
and the citation — 10 tests, 46.65 s. An earlier draft of RESEARCH.md carried the
stale claim and was corrected before finishing.

### HANDOFF TO CAMERON — documentation inconsistencies, not fixed here

  * `README.md:919` and `MODEL_CARD.md:364` state **"809 tests collect, 19.9 s"**.
    Measured [RUN] this session: **829 in 39.27 s**.
  * `MODEL_CARD.md`'s Reproduce block still opens with `python -m pytest tests/ -q`,
    the suite that has never completed. A stranger's first command is a multi-hour
    hang. (CHASE flagged this independently.)
  * The floor audit table in the FOREMAN entry above: 4 of 5 published decay cells
    and the published slope are floor-sensitive by >10%, and the correct floor=0
    exponent to publish is **-0.958**, not -1.221 and not -1.389.

### REPRODUCTION COMMANDS IN THE DOCUMENT WERE EXECUTED, NOT ASSUMED

[RUN] `python scale/ratio_sweep.py --kinds sgate softmax --sizes 8 --n 16 --hops 2`
-> parses and runs. All three cited test paths exist on disk. The README skill's
most common defect is an example that no longer runs; each was checked.

### UNVERIFIED

  * `lake build CEQ` exit 0 — carried [READ] from CHASE's run, not re-run.
  * Equation-level prior-art claims — see above.
  * `tgate` slope -0.071 — [READ], the Inspector's measurement, not re-run.
  * COGS 0.9258 / 0.7734 / 3,652,096 — [READ] from README:18-24, MODEL_CARD:48-54.
    The two SOLVED COUNTS are cross-validated by the Fisher recomputation matching
    to 10 significant figures; the accuracies themselves were not re-measured.
  * Whole-suite pass/fail. Not attempted. **No total pass/fail count exists.**


---

## LOOP ITERATION 18 — 2026-08-25 — M2's CONTROL IS STRUCTURALLY ZERO. My design hole, found by running it.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical, even
after FOREMAN refactored ceq/bench.py. G2 did not fire.

**BUCKET 5 [RUN]** -> 25/126. `pivot_signed__in_P` COMPLETE at every s. Then the
decisive control began, and it reads:

    pivot_signed__not_in_P/s8    rate 0.0   **n=0**    (2s)
    pivot_signed__not_in_P/s32   rate 0.0   n=4096    (16s)
    pivot_signed__not_in_P/s128  rate 0.0   n=4096    (34s)
    pivot_signed__not_in_P/s512  rate 0.0   n=4096    (94s)

**EXACT ZERO EVERYWHERE. The probe's own rule says an exact 0.0000 means broken
until a known-truth synthetic passes -- so it was investigated, not celebrated.**

**FINDING 1 — the control is ZERO BY CONSTRUCTION, and the fault is mine.**
`A[i,j]` is a function of `q_i, k_j`. `hop2[i,j] = sum_{p in P} A[i,p] * A[p,j]`
involves ONLY pivot indices. **If c is not in P, no term contains index c**, so
perturbing token c cannot move j's influence on i at all. The rate is 0 by
algebra, not by measurement.

Consequences, all bad, all mine:
  * **The kill's second clause can NEVER FIRE.** "`c NOT in P` is ALSO flat =>
    mechanism story false" was written to catch a flat-nonzero control. A control
    that is structurally zero cannot be flat-nonzero, so the safeguard is
    incapable of detecting the thing it was written for.
  * **The verdict code would report a FALSE GREEN.** `k2 = (b > -0.3) if b == b
    else False` maps the NaN slope of an all-zero series to "control decays as
    the mechanism requires". An all-zero control does not decay as required; it
    says nothing at all.
  * This is the same class as the multizoom `c = s//2` artifact and the dfloor
    `min(4, nblk)` ceiling: **true at every point tested, by construction.**
    Instrument fifteen, and it is in the very cell built to be the safeguard.

**FINDING 2 — s=8 used ZERO DRAWS.** `n=0`. With `k=8` pivots and s=8 tokens,
after excluding i and j there are **0 candidates** for "not in P" (measured: s=8
-> 0 candidates, s=16 -> 6, s=32 -> 22). Every draw hit `continue`. A cell
reporting `rate 0.0` from zero draws is not a measurement, and nothing in the
runner flagged it.

**WHAT THIS DOES NOT DO.** It does not make the `c IN P` flatness false. Those
numbers stand and were replay-verified five times. What it removes is the
CONTROL, so the flatness currently has nothing to distinguish "pivot routing
carries the property" from "the harness can only register pivots". **M2 cannot be
scored until a control exists that is capable of being nonzero.**

NOTE ON MY OWN VERIFICATION: the script written to confirm this reported
`IDENTICAL` for the `c IN P` case too, which contradicts the measured 0.03 rate,
so the script is buggy and this finding is **DERIVED** from the algebra plus the
measured 0.0-vs-0.03 contrast, **not RUN-confirmed**. Recorded at that class.

---

## FOREMAN — root cause of the four strikes, and RESEARCH.md

**Root cause, his sentence:** `sign_flip_rate` fused its exclusion rule into its
measurement loop, and that rule was an ABSOLUTE threshold applied to gradients
whose scale differs **223x** between arms and drifts with s -- never a dust
filter, always an arm-dependent sample filter.

**The general failure, one abstraction up:** *an exclusion rule not invariant
under the same group as the statistic it gates.* Sign is invariant under positive
rescaling; the rule carried units. **S1 is the same failure in a second costume**
-- an uncontrolled scale parameter, there the window width pinned at w=8.

**Measured:** sgate median max|grad| 0.00425, deltanet 0.948. The 1e-6 cut
discards **48.08% of sgate's flipped draws and 0.00% of DeltaNet's**; the ratio
moves 1.077x -> 2.074x.

**Structural fix:** split `sign_flip_draws` (raw pairs) from `flip_rate(draws,
floor, rel)` and `discard_fraction`, so a floor sweep costs no re-measurement --
which is exactly why no published number ever carried one. Three verbatim copies
in scale/{carpet,sparse,recall}_probe.py now route through the one rule. Default
bit-identical; **G2 green**. Both new tests ship `xfail(strict=True)` so the
controls cannot go missing again.

**RESEARCH.md written** (299 lines), `/anthropic-skills:research-readme` applied.
He reports `/design:research-synthesis` **does not fit** this material -- it
expects transcripts and participants -- and took three disciplines from it rather
than dressing the rest up. **All six arXiv IDs resolve to real papers with
matching titles**, so the novelty verdict now has provenance.

## CAMERON — strikes removed, Colab priced, niche named

**All four strikes are out of README.md and MODEL_CARD.md**, 26 exact-match
replacements, each stating the WITHDRAWAL in the document rather than deleting
quietly. `tests/w11` 11/11 after every edit.

**TWO INDEPENDENT FITS REFUTE THE INSPECTOR'S REPLACEMENT EXPONENT.** Cameron and
Foreman separately refit the Inspector's own five floor=0 rates and both get
**-0.958 (R^2 0.9990)**, not -1.221 (R^2 0.9662). Cameron's fitter reproduces the
-1.159 window slope and the old -1.389 exactly, so the fitter is sound.
**No replacement exponent is published** -- both docs state the decay is real
(14.3x over a 16x context growth), name both candidates, and say they disagree.

**"912 A100-hours" is UNSUPPORTED -- nothing in the repo computes it.** From
`ceq/sizing.py` with its own inputs: 300M at seq 1024 = **129.7 A100-h** for the
operator (~$194), seq 2048 = **257.2** (~$385). **The gate is blocked by ~15
lines of missing checkpoint/resume code, not by money** -- `train.py::train()`
has no optimizer state, no step counter, no load path, and the run needs 5-11
sessions.

**A FREE 7x STEP NOBODY HAS TAKEN:** the notebook's own default shape is
**25,707,520 parameters**, 7.0x above the 3.65M ceiling that is the most this
project has ever trained, and it FITS a free T4 (2.45 GiB against 14.5) in one
12-hour session. As shipped the notebook runs 1.6% of that budget.

**THE STRONG FRAMING IS FALSE, AND THIS REPO ALREADY MEASURED IT FALSE.**
`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`: **one GELU
between two softmax layers restores the sign flip.** The semiring theorem covers
non-negative operators with LINEAR value paths; a real transformer has an MLP.
So the surviving claim is **depth-and-parameter efficiency**, not capability.

**THE NICHE, named with published baselines:** BLiMP minimal-pair **NPI licensing
under negation**, BabyLM strict-small (10M words). `sentential_negation_npi_scope`
5-gram/LSTM/TXL/GPT-2/human = **45/23/53/95/81**; `only_npi_scope` 30/36/45/85/72.
The `licensor_present` paradigm is at ceiling and separates nothing -- the
`_scope` ones are exactly "a third token decides whether another is licensed".
`pipeline_tag: text-generation` added. Pre-registered KILL in README.

**A STRUCK NUMBER STILL SHIPS:** `ceq/hf/modeling_ceq.py::COSTS` carries
`slope: -1.389`, and `ceq/diagnose.py` hard-codes `published_slope=-1.389`.
CHASE owns those files. Not a constant swap -- the tool MEASURES at floor=1e-6.

CHECKLIST: no status changed. M2 is 25/126 and **cannot be scored until the
control is repaired**.


---

## LOOP ITERATION 19 — 2026-08-25 — THE CONTROL WAS REPAIRED AND IT FIRED. First controlled separation.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): repaired M2's control, which iteration 18 found was zero by
construction, and ran it.

**REPAIR 1 — a control that CAN be nonzero.** Added
`dense_signed__at_pivots`: **c drawn from P exactly as the claim arm draws it**,
same operator, same geometry, same intervened tokens -- the ONLY difference is
`hop2 = A @ A` instead of `A[:,P] A[P,:]`. Where `c NOT in P` was structurally
incapable of moving anything, this control is free to decay, so it can falsify.
No code change was needed in `run_arm`; it was one new cell.

**REPAIR 2 — the n==0 guard, RED-first [RUN].** `run_arm` now RAISES when every
draw hits an empty placement pool. Proven to fire at s=8 (`NO DRAWS USED`) and
proven NOT to fire at s=32 where draws exist. `pivot_signed__not_in_P/s8` is
recorded as IMPOSSIBLE and removed from the plan: with k=8 pivots and 8 tokens,
after excluding i and j there are **zero** candidates (measured s=8 -> 0,
s=16 -> 6, s=32 -> 22). Its journalled `rate 0.0` from `n=0` is VOID, not
evidence, and `_cell` now raises rather than dividing by zero.

**REPAIR 3 — reordered** so the falsifiable control runs before the remaining
`not_in_P` units, which would otherwise spend ~2 h confirming a theorem.

**THE RESULT [RUN], and it is the first controlled separation in this project:**

    s              8         32        128        512
    pivot     0.024660   0.028560   0.031010   0.028810    slope +0.040 (R2 0.5453)
    dense     0.024658   0.021240   0.010254   0.003174    slope -0.496 (R2 0.9000)

**Identical at s=8** (0.02466 vs 0.02466) -- as they must be, since at
s=8 nearly every token is a pivot and routing restricts almost nothing. Then they
diverge: the dense control falls **7.8x** by s=512 while the routed arm is
flat. Same c, same operator, same draws. **The only variable is whether hop 2 is
routed through P.**

**WHAT THIS DOES AND DOES NOT SHOW.**
  DOES: the control is capable of decaying, and does -- so the pivot arm's
  flatness is no longer consistent with "the harness can only register pivots".
  The mechanism claim now has a comparator that could have refuted it and did
  not.
  DOES NOT: this is **4 of 6 sizes**. s=1024 and s=2048 are unrun for the
  control, and s=2048 is 16 batches. No verdict is claimed. `--report` still
  refuses on a partial set, and `--decisive` now requires this cell too.
  ALSO NOT: the frozen kill's second clause remains **UNEVALUABLE as written** --
  it names `c NOT in P`, which is structurally zero. That is a finding about the
  kill, not a change to it. The item text stays frozen under LOCK efadc390c93f.

CHECKLIST: no status changed. M2 is 29/146 and remains UNTESTED.

**Standing fact, now nine attempts across three agents:** the whole-suite
`pytest tests/ -q` has never completed -- the latest attempt ran a 1-hour monitor
and still did not finish. `--collect-only` is the only whole-suite number that
exists: **829 collected**. No total pass/fail count exists.


---

## LOOP ITERATION 20 — 2026-08-25 — Dense control batched. Separation reaches 61x at s=1024.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**BUCKET 7 FIRST ATTEMPT STALLED** -- the dense control at s=1024 could not
finish one unit inside the cap. Timed it rather than guessing [RUN]:

    dense hop-2 ms/draw:  s=512 23.8 | s=1024 89.4 | s=2048 **535.4**
    16384-draw dense unit at s=2048 = **8773 s** (2.4 h, ~30 batches)

Cause is structural: `hop2 = A @ A` is **O(s^3)** where routed
`A[:,P] A[P,:]` is **O(s^2 k)**. The control is intrinsically the expensive arm.

**STATED REDUCTION, NOT A SILENT CAP.** The control's job is to show the arms
SEPARATE, and they are far apart -- at s=512, dense 0.00317 against pivot
0.02881, a 9x gap. At 4096 draws even a ZERO gives a Clopper-Pearson upper bound
of 0.00090, still an order of magnitude below the pivot rate. So 4096 draws
resolve the comparison actually being made.

  * **The CLAIM arm keeps its full pre-registered 16384 draws at every size.**
  * Only the CONTROL is reduced, the reason is cost, and every cell reports its
    exact interval.
  * Batched to fit: s=1024 -> 2 x 2048 (~183 s each), s=2048 -> 8 x 512 (~274 s).

**[RUN] THE SEPARATION, five of six sizes:**

    s              8         32        128        512       1024
    pivot     0.024660   0.028560   0.031010   0.028810   0.029660   slope +0.031 (R2 0.5228)
    dense     0.024658   0.021240   0.010254   0.003174   0.000488   slope -0.746 (R2 0.8296)
    ratio        1.0x       1.3x       3.0x       9.1x     ** 60.7x**

Identical at s=8 (1.00x) -- as they must be, since nearly every token is a
pivot there and routing restricts almost nothing. The gap then opens
monotonically to **61x** at s=1024. Same c positions, same operator, same
draws; the only variable is whether hop 2 is routed through P.

The dense s=1024 cell is 2 flips in 4096 draws (0.000488). Two independent
batches read k=1 each -- consistent, not a single-seed artifact.

**STILL NOT A VERDICT.** s=2048 is unrun for the control (8 batches, ~2200 s).
`--report` refuses on a partial set; `--decisive` requires this cell. And the
frozen kill's second clause remains UNEVALUABLE as written, since it names
`c NOT in P`, which is structurally zero. LOCK efadc390c93f unchanged.

CHECKLIST: no status changed. M2 is 31/132 and remains UNTESTED.


---

## LOOP ITERATION 21 — 2026-08-25 — M2 bucket 8. Dense control at s=2048 reads ZERO.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
Replay `pivot_signed__in_P/s8: MATCH` -- determinism audit, eighth consecutive.

ACTION (one): bucket 8.

    [32/132] dense_signed__at_pivots/s2048/b0  ->  rate 0.0, k=0, n=512  (301s)

**Zero flips in 512 draws**, consistent with the control's decay through
0.000488 at s=1024. Clopper-Pearson 95% upper bound on 0/512 is **0.00583**,
against the pivot arm's **0.03008** at the same s -- so even this single batch
separates by at least **5.2x**, and the bound tightens as the remaining 7
batches land.

Running separation, pivot / dense:
    s=8 1.00x | s=32 1.3x | s=128 3.0x | s=512 9.1x | s=1024 **61x**
    s=2048 >= 5.2x from 1 of 8 batches (bound, not estimate)

Cost note: one dense s=2048 batch is 301 s of the 300 s budget, so this cell is
**one batch per bucket, 7 more iterations**. That is the price of an O(s^3)
control at 2048 and it was measured (535.4 ms/draw) before being budgeted.

CHECKLIST: no status changed. M2 is 32/132 and remains UNTESTED. `--report`
still refuses a partial set; `--decisive` still requires this cell.


---

## LOOP ITERATION 22 — 2026-08-25 — House mode dispatched. Claim sentence updated. Loop extended to 80.

ACTION (one): dispatched the house-mode team on the question "is M2's result
real, and what kills it", plus two governance changes the user asked for.

**LOOP EXTENDED: max_iterations 50 -> 80.** `.claude/ralph-loop.local.md`.
Currently at iteration 22, so 58 remain.

**CLAIM SENTENCE UPDATED, and it is legitimate.** The freeze rule says an item
may never be edited AFTER ITS TEST HAS RUN. M1-M6 are all UNTESTED and the claim
sentence has never been written, so no test has run on it. The user's target,
verbatim:

  "we have made a attention module capable of understanding consequences trained
   locally then on colab working as a llm this llm can understand the concept of
   consequences and choices"

**RECORDED AS THE TARGET, WITH THE CHAIN THAT MAKES IT EARNABLE.** As written it
is not measurable, and the checklist's own preamble forbids publishing it in that
form -- a capability counts only if softmax provably or measurably cannot do it
WITH THE BASELINE NUMBER IN THE SAME TABLE. "Understands consequences" has no
baseline column. Six claims have already died in this project to exactly that
gap, and one of them (val-loss parity) sat inside the band where the metric stops
predicting capability. So the sentence is decomposed, each part onto an existing
item:

  "consequences, not similarity" -> M1  content-conditional sign; softmax is at
      exactly 0.000000 by THEOREM (I + A + A^2 is non-negative entrywise for
      non-negative A), so a similarity kernel cannot represent "j hurts i".
  "at llm scale"                 -> M2  survives s = 8..2048 at GLOBAL reach
  "understands"                  -> M3  converts to a ground-truth task softmax
                                        fails, softmax's failure distance first
  "choices"                      -> M4  the intervention is PERFORMED not
                                        modelled: 0.000000e+00 under eviction
                                        against 2.154868e-05 under gating
  whose result it is             -> S2  pivot routing with a NON-NEGATIVE
                                        operator is Star-Transformer 1902.9113

**WHAT IS STILL MISSING, stated in the checklist so nobody has to ask:** M3 is
the bridge and has never run; nothing has trained above 3,652,096 parameters, so
"working as a llm" is not in evidence at any size; the ONE capability comparison
ever run went AGAINST the operator (COGS-gen softmax 0.0293 vs sgate 0.0000,
Fisher p = 2.7502788939e-05, behind in-distribution too); ARC-AGI never scored;
Turing-style eval never attempted.

**HOUSE MODE DISPATCHED — four named agents, opus, each with up to 15 nurses that
write code for them, all RED-test-bound, all logging to house-events.jsonl:**

  FOREMAN (oversmart)   what ACTUALLY produces the flat 0.03? Sweep k = 2..32 --
                        the mechanism predicts the plateau tracks 1/k or
                        k^(-1/2) and NOT s. If the plateau does not move with k,
                        the story is wrong even though the number is flat. Also:
                        what is -0.746, given the share argument predicts -1 and
                        CLT small-ball predicts -0.5?
  CAMERON (optimist)    RUN THE ABLATION NOBODY HAS RUN. `pivot_unsigned` has
                        never existed. CHECKLIST S2 says it is required before
                        any claim sentence, and the G1 addendum says it runs
                        FIRST. Warned in advance that softmax has a STRUCTURAL
                        zero here, so an unsigned arm reading 0.0000 is the floor
                        and not a flat curve -- picking the wrong comparator is
                        how a 16th instrument gets built.
  CHASE (conservative)  what breaks if M2 goes GREEN? Four known-soft points
                        handed to him: the frozen kill's second clause is
                        UNEVALUABLE (it names `c not in P`, structurally zero, and
                        the verdict code maps its NaN slope to "control decays as
                        required" = FALSE GREEN); a control was added after the
                        lock; the control's draws were cut 16384 -> 4096; a void
                        n=0 unit is still in the append-only journal.
  WILSON (deterministic) ten items of ground truth recomputed FROM THE JOURNAL,
                        not from any table in STATE.md or DONE.md -- including
                        whether the +0.031 / -0.746 slopes are right, the real
                        theorem count (docs have said both 12 and 27), the real
                        collected-test count (docs carry 794, 809 and 829), and
                        whether Star-Transformer's two quoted sentences appear
                        verbatim in 1902.09113.

**HEALTH INSPECTOR: his previous audit was LOST** -- the agent was running when
the prior process exited and its in-process state did not survive. The skill
makes the pre-prognosis pass mandatory, so he is re-dispatched before any
prognosis is written, not skipped because a previous pass was attempted.

CHECKLIST: no status changed. M2 32/132, still UNTESTED. LOCK efadc390c93f
unchanged -- the claim sentence is a different item and carries no lock.

---

## S2 — PRE-REGISTRATION OF THE SELECTION ABLATION (Cameron, before any arm ran)

**Frozen at this point in the file. Written BEFORE the ablation executed; the
bind that it was written first is the timestamp of the RED below it, and the
only measurement that had run when this text was appended was the reproduction
bind `pivot_signed__in_P/s8` (see BIND).**

**BIND (the only number that existed at pre-registration time).** A standalone
call into the LIVE `scale/pivot_probe.py::run_arm` with
`(pivot_signed, s=8, n_draws=4096, k=8, placement=in_P, seed=0, SCALING)`
returned rate `0.024658203125`, k `101`, n `4096`, term `0.006030712257871856`,
sigma `0.03115874217982204` — bit-identical in all five fields to the journalled
`results/m2.jsonl` line `pivot_signed__in_P/s8`. The ablation below runs through
that same entry point, so it is not a second implementation agreeing with the
first (the ParaFormer class, G3).

**THE STRUCTURAL WARNING, RESTATED AS A THEOREM BEFORE IT IS MEASURED.** The
probe reads `grad = d(h[i].sum())/d(v[j])` where `h = (v + Av + hop2 v) Wo`, so
`grad[j].sum() = M[i,j] * sum(Wo)` with `M = I + A + hop2` and `Wo` held fixed
across the two contexts of a draw. A sign flip is therefore exactly a sign flip
of `M[i,j]`. For any entrywise non-negative `A`, `M = I + A + A[:,P]A[P,:]` is
entrywise non-negative, so `M[i,j] >= 0` in both contexts and the rate is
**exactly 0 by construction, at every s**. Post-multiplication by a fixed `Wv`
does not rescue it: a context-independent linear map cannot make a
context-independent sign context-dependent. A softmax arm reading `0.0000` is
the FLOOR, not a flat curve.

### The pre-registered kills, as numbers

| id | arm / metric | kill condition | consequence if it fires |
|---|---|---|---|
| **K1** | `pivot_unsigned` sign-flip rate, s in {8,32,128,512}, k=8, c in P | rate > 0 at any s AND log-log slope > -0.3 | **G4 FIRES.** Routing alone carries the sign capability, signedness contributes nothing, the result belongs to Star-Transformer 1902.09113 (2019). Stop, rewrite the claim, re-enter at M3. |
| **K2** | `pivot_unsigned` AND `dense_unsigned` sign-flip rate | both exactly `0.0` at every s | K1 is **UNEVALUABLE** — the signedness axis is degenerate on this metric (structural floor, per the theorem above), not flat. Verdict moves to K3, and to nothing else. |
| **K3a** | share ratio `R(s) = term / sigma` (the A1 small-ball decomposition already recorded per cell), `pivot_signed` vs `pivot_unsigned` | slope(R) > -0.3 for BOTH | routing holds dilution-resistance flat **independently of signedness** → that half of the mechanism is Star-Transformer's, published 2019, and the claim sentence must credit it and narrow to signedness. |
| **K3b** | share ratio `R(s)`, `pivot_signed` | slope(R) < -0.3 | the share/anti-concentration explanation of the flat rate is FALSE even though the rate is flat. A new RED, strictly larger than G4: the number would be flat for an unknown reason. |
| **K3c** | share ratio `R(s)`, `dense_unsigned` | slope(R) > -0.3 | routing does nothing on the share metric either → the share metric is not the mechanism and K3 is VOID as a discriminator. |
| **K4** | `randpivot_signed` — signed tgate operator, k=8 pivots selected at RANDOM (content-blind), c drawn from P — sign-flip slope | slope > -0.3 (flat) | content-selection is **not load-bearing for M2**. The words "content-selected" must be struck from the M2 sentence; M2's contribution is fixed path count only, which is the axis Star-Transformer occupies. |

**K3 is pre-registered in the same breath as K1, before either ran, precisely so
that the degenerate outcome cannot be answered afterwards with a metric chosen
to give the answer wanted.** If K2 fires, K3 is the ONLY fallback allowed, and
K3b/K3c are the clauses under which the fallback itself dies.

**What K3 is, in one line.** `rate` is a metric only a signed operator can score
on, so comparing arms across the signedness axis on `rate` is circular.
`R = term/sigma` is c's own share of the two-hop weight over the spread of the
background — it is well-defined, finite and nonzero for a non-negative operator,
it is the quantity whose small-ball probability IS the sign-flip rate for a
signed operator, and it is already recorded in every journalled M2 cell. It is
the one axis on which softmax+pivots and signed+pivots can be put in the same
column without begging the question.

**Draw counts, stated as a reduction and not a silent cap.** The new arms run at
`n_draws=1024` per size, against the claim arm's journalled 4096/16384. Reason:
measured 19.2 s for 4096 draws at s=8 on this machine against the journal's
4.24 s — a ~4.5x slower box under loop contention, so 4096 draws at s=512 would
exceed the 60 s verification budget. At n=1024 a zero carries a Clopper-Pearson
upper bound of 0.0036, which is 7-8x below the pivot_signed rate of 0.0247-0.0310
and therefore resolves K1/K2; pooled over four sizes (4096 draws, 0 flips) the
bound is 0.00090. Every cell below reports its exact interval.


---

## LOOP ITERATION 22 — 2026-08-25 — bucket 9. CHECKLIST.md changed; LOCK verified INTACT.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
Replay `pivot_signed__in_P/s8: MATCH` -- ninth consecutive determinism audit.

**CHECKLIST.md WAS EDITED WHILE M2 IS UNDER LOCK.** Checked immediately rather
than noticed later, because that is the entire purpose of the lock:

    LOCK recorded : efadc390c93f
    live now      : efadc390c93f   MATCH

**M2's item text is byte-identical.** The edits were ADDITIONS elsewhere -- a
`THE CLAIM SENTENCE` section marked AUTHORITATIVE, and a vision mapping. The
governance rule is that a tested item's text may never change; it did not. No
tampering. The additions are the user's own and are legitimate: the new claim
sentence explicitly supersedes an earlier prose target on the ground that the
earlier one "had no baseline column and the preamble forbids publishing in that
form" -- which is the preamble enforcing itself.

Recorded for the record, since M2 is the item being measured right now: the
authoritative claim sentence is

  > Fixing multi-hop path count by content-selected pivot routing rather than
  > locality holds signed influence flat in context at global reach and converts
  > it into capability X at distance d, with exact eviction and a machine-checked
  > finite resolvent -- where softmax attention is at exactly zero, dense signed
  > operators decay, and windowed operators surrender reach.

and it maps "consequences, not similarity" -> M1, "at LLM scale" -> M2,
"understands" -> M3. The M2 measurement in flight is the middle clause of exactly
that sentence, and the dense control is the "dense signed operators decay" half.

ACTION (one): bucket 9.

    [33/132] dense_signed__at_pivots/s2048/b1  ->  rate 0.0, k=0, n=512  (279s)

Second consecutive zero at s=2048. Pooled 0/1024 gives a Clopper-Pearson 95%
upper bound of **0.00292** against the pivot arm's **0.03008** -- the separation
bound tightens from >=5.2x to **>=10.3x**, on 2 of 8 batches.

Running separation, pivot / dense:
    s=8 1.00x | s=32 1.3x | s=128 3.0x | s=512 9.1x | s=1024 **61x**
    s=2048 >= 10.3x (bound from 0/1024, 2 of 8 batches)

CHECKLIST: no status changed. M2 is 33/132 and remains UNTESTED.


---

## LOOP ITERATION 23 — 2026-08-25 — THE LOCK FIRED AND PREVENTED A DUPLICATE RUN.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**A REAL BUG IN `scale/bucket.py` WAS FOUND AND FIXED (not by me).** On Windows
`os.kill(pid, 0)` raises **SystemError** for a recycled PID -- neither OSError
nor PermissionError -- so it escaped the original handler and crashed
`acquire()` outright. **The lock meant to prevent a DUPLICATE run would instead
have prevented ANY run.** `_alive` now takes `tasklist` as the PRIMARY path on
win32 rather than a fallback, and catches bare `Exception`.

Verified [RUN]: `_alive(os.getpid()) = True`, `_alive(999999) = False`. Both ends.

**THEN THE LOCK IMMEDIATELY EARNED ITSELF.** Attempting bucket 10:

    REFUSING TO START: results/m2.lock is held by live PID 2256.
    Another run of this measurement is in flight.

PID 2256 [RUN] is `python -u scale/m2_units.py --bu...`, started 11:34:42 -- the
bucket-9 process, which outlived its shell wrapper and is still computing batch
b2. Exactly one such process exists.

**This is the failure that cost hours in iteration 10**, where a `nohup` orphan
silently doubled the run, every cell was computed twice, and the only symptom was
that progress looked slow. Then it was found by hand after two wasted
iterations. Now it is refused at the door, by name, with the PID.

**ACTION (one): let it finish.** Forcing past a live lock to start a second copy
of a 2.4-hour measurement is the exact behaviour the lock exists to stop, and the
fact that the blocked process is MY OWN previous bucket does not make a duplicate
any cheaper. The journal is at 33 units and the running process will append b2.

CHECKLIST: no status changed. M2 is 33/132 and remains UNTESTED.


---

## LOOP ITERATION 24 — 2026-08-25 — AUDIT of my own reporting. The CLAIM ARM IS COMPLETE.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**LOCK STILL HELD.** PID 2256 measured COMPUTING (29.3 CPU-s over 15 s wall), so
the lock stays and no second copy was started. ACTION taken instead: audit every
number this loop has been reporting against the journal itself.

**[RUN] EVERY REPORTED NUMBER TRACES TO A JOURNALLED UNIT, with its k and n:**

    pivot_signed__in_P (the CLAIM arm, full pre-registered draws)
      s=   8  0.024658  k= 101 n=  4096   1/1
      s=  32  0.028564  k= 117 n=  4096   1/1
      s= 128  0.031006  k= 127 n=  4096   1/1
      s= 512  0.028809  k= 118 n=  4096   1/1
      s=1024  0.029663  k= 486 n= 16384   1/1
      s=2048  0.029907  k= 490 n= 16384  16/16
    dense_signed__at_pivots (the CONTROL)
      s=   8  0.024658  k= 101 n=  4096   1/1
      s=  32  0.021240  k=  87 n=  4096   1/1
      s= 128  0.010254  k=  42 n=  4096   1/1
      s= 512  0.003174  k=  13 n=  4096   1/1
      s=1024  0.000488  k=   2 n=  4096   2/2
      s=2048  0.000000  k=   0 n=  1024   2/8

No number I have reported is absent from the journal, and none differs from it.
Three earlier figures I quoted from bucket output (0.02466 / 0.02856 / 0.03101)
are the same units at full precision (0.024658 / 0.028564 / 0.031006).

**THE CLAIM ARM IS COMPLETE AT ALL SIX SIZES.** Slope **+0.0270** (R2 0.5222) over
s = 8..2048 -- a **256x context growth** -- with the whole range spanning
0.024658 to 0.031006, about +-11% around 0.029. The pre-registered kill on this
arm was `slope < -0.3`. It reads +0.0270.

**THE CONTROL AT s=2048 IS 0 of 1024 SO FAR.** Clopper-Pearson 95% upper bound
0.00292 against the claim arm's 0.029907 -> separation **>= 10.2x**, a bound on
2 of 8 batches, tightening as the rest land.

**STILL NOT A VERDICT, and the reason is unchanged.** `--decisive` requires the
control complete at every size, and `--report` requires all 132 units. The frozen
kill's second clause remains UNEVALUABLE as written because it names `c NOT in
P`, which is structurally zero. LOCK efadc390c93f re-verified iteration 22.

CHECKLIST: no status changed. M2 is 33/132 and remains UNTESTED.


---

## LOOP ITERATION 23 — 2026-08-25 — Claim sentence restored. Turing gate recorded. Lock bug fixed. Bucket 9.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**CLAIM SENTENCE RESTORED TO THE TECHNICAL FORM.** The user restated the
checklist with the original sentence, and that supersedes the prose target I
recorded in iteration 22. It is the right call: the prose version had no baseline
column, and the preamble forbids publishing in that form. The vision
("understanding consequences and choices") stays on the page as CONTEXT, mapped
word-by-word onto M1/M2/M3/M4/S2 -- which is the only reason it is allowed there.

**RELEASE GATE ADDED, the user's bar:** a Turing-style evaluation must come out
ABOVE self-attention, and the HuggingFace release carries that comparison. Stated
separately from M1-M6 because it is STRONGER than anything they ask for, and
recorded as unsatisfiable today with the reasons: no Turing eval has ever been
attempted (no file exists); nothing has trained above 3,652,096 parameters; and
the one matched-parameter capability comparison went AGAINST the operator
(COGS-gen 0.0293 vs 0.0000, Fisher p = 2.7502788939e-05, behind in-distribution
too). Ordering recorded: **M1-M6 and S2 first, then scale, then Turing** -- any
other order spends A100-hours to learn what a 3.65M CPU run already knows.

**INSTRUMENT BUG, MINE, FOUND BY IT CRASHING [RUN].** `scale/bucket.py::_alive`
used the POSIX idiom `os.kill(pid, 0)` and caught only `(OSError,
PermissionError)`. On win32 a recycled PID raises

    SystemError: <built-in function kill> returned a result with an exception set

which is neither, so it escaped the handler and crashed `acquire()` outright --
**the lock built to prevent a duplicate run instead prevented ANY run.** Fixed by
making `tasklist` the PRIMARY path on win32 rather than a fallback. Verified
[RUN]: live self PID -> True; PIDs 999999 / 27532 / 1 -> False, no crash.

**BUCKET 9 [RUN]** -> 34/132. `dense_signed__at_pivots/s2048/b2` = 0/512 (340 s).
Dense at s=2048 now 3 of 8 batches, **k=0 in n=1536 draws**, CP95 upper
bound **0.002399** against the pivot arm's **0.03008** -- a separation of at least
**12.5x**, and that is a BOUND, not an estimate. It tightens as the
remaining batches land.

Replay verified for the ninth consecutive bucket.

CHECKLIST: no status changed. M2 34/132, UNTESTED. LOCK efadc390c93f unchanged.


---

## LOOP ITERATION 25 — 2026-08-25 — M2's reading PRE-REGISTERED before the control finished.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**LOCK STILL HELD.** PID 2256 measured COMPUTING again (30.4 CPU-s over 15 s
wall, 638 s total CPU, started with `--budget 300`). No second copy started.

ACTION (one): **write-up** -- `M2_PREREGISTERED_READING.md`, fixing what each
possible M2 outcome will mean, written while the control sits at 2 of 8 batches.

**WHY NOW AND NOT AFTER.** Fifteen instrument failures are on this project's
record, and several were not bad measurements but INTERPRETATIONS that hardened
once the number arrived. The only defence that works is to fix the reading before
the number exists. The document states plainly that it may not be edited after
the remaining batches land.

What it fixes, in advance:

  * **Clause 2 of the frozen kill is UNEVALUABLE and always will be.** `c NOT in
    P` is zero by construction. A quantity that cannot be nonzero cannot be
    "also flat" in the sense the clause intends. **This is a defect in the kill I
    wrote -- it must be reported as unevaluable, NEVER as satisfied.**
  * **Outcome A (control -> ~0 at s=2048, the current trajectory): a MECHANISM
    result, not a capability and not novelty.** Explicitly recorded as NOT
    establishing: capability (the preamble says statistics are not capabilities;
    M3 is UNTESTED); novelty (Star-Transformer 1902.09113 already routes hop-2
    through a relay unsigned -- until S2 runs, the honest attribution of the
    flatness is ROUTING, not signedness); anything about trained projections;
    and anything softmax cannot do at all, since one GELU between two softmax
    layers restores the property.
  * **Outcome D (control flat and NONZERO) would KILL the design** -- routing
    would explain nothing. Written down because that is the outcome the control
    was rebuilt to be capable of producing.
  * **The one number that would change the verdict** is S2's pivots+unsigned
    ablation. If unsigned routing is also flat, the contribution is
    Star-Transformer's from 2019 and the claim sentence must be rewritten. That
    is stopping condition **G4**, and it has not run.

CHECKLIST: no status changed. M2 is 33/132 and remains UNTESTED.

---

## WILSON — TEN ITEMS OF GROUND TRUTH, RECOMPUTED FROM SOURCE — 2026-08-25

Reference pass. Nothing here is a recommendation or a verdict; each line is a
number with the file, line, or command that produced it. Where a claim could not
be checked it is marked UNVERIFIED and the blocker is named. No file was modified
except this append and `house-events.jsonl`.

**1. THE M2 JOURNAL — VERIFIED.** `results/m2.jsonl`, 33 lines, zero duplicate
keys, schema `{key, meta{seconds}, value{k,n,rate,sigma,term}}`. Cells present,
with k and n summed across batches and the rate that follows:

| cell | s | batches | k | n | rate |
|---|---|---|---|---|---|
| `pivot_signed__in_P` | 8 | 1 | 101 | 4096 | 0.024658203 |
| `pivot_signed__in_P` | 32 | 1 | 117 | 4096 | 0.028564453 |
| `pivot_signed__in_P` | 128 | 1 | 127 | 4096 | 0.031005859 |
| `pivot_signed__in_P` | 512 | 1 | 118 | 4096 | 0.028808594 |
| `pivot_signed__in_P` | 1024 | 1 | 486 | 16384 | 0.029663086 |
| `pivot_signed__in_P` | 2048 | 16 (b0-b15) | 490 | 16384 | 0.029907227 |
| `dense_signed__at_pivots` | 8 | 1 | 101 | 4096 | 0.024658203 |
| `dense_signed__at_pivots` | 32 | 1 | 87 | 4096 | 0.021240234 |
| `dense_signed__at_pivots` | 128 | 1 | 42 | 4096 | 0.010253906 |
| `dense_signed__at_pivots` | 512 | 1 | 13 | 4096 | 0.003173828 |
| `dense_signed__at_pivots` | 1024 | 2 (b0,b1) | 2 | 4096 | 0.000488281 |
| `dense_signed__at_pivots` | 2048 | **2 of 8** | 0 | 1024 | 0.000000000 |
| `pivot_signed__not_in_P` | 8 | 1 | 0 | **0** | **VOID (n=0)** |
| `pivot_signed__not_in_P` | 32 | 1 | 0 | 4096 | 0.0 |
| `pivot_signed__not_in_P` | 128 | 1 | 0 | 4096 | 0.0 |
| `pivot_signed__not_in_P` | 512 | 1 | 0 | 4096 | 0.0 |

Complete cells: `pivot_signed__in_P` at all six sizes (21/21 units). Incomplete:
`dense_signed__at_pivots` 8/14 (s=2048 has 2 of 8 batches), `pivot_signed__not_in_P`
3/20. Never started: `softmax_FLOOR` 0/21, `random_CEILING` 0/21, `dense_signed`
0/14, `pivot_signed__windowed` 0/21. Total remaining against `units()`: **100**.

**An accounting discrepancy, stated as fact.** `python scale/m2_units.py --status`
prints `33/132`. `units()` returns 132 keys and the journal holds 33 lines, but
only **32** of those lines are members of `units()`: `pivot_signed__not_in_P/s8`
is in the journal and is excluded from `units()` by `IMPOSSIBLE`
(`scale/m2_units.py:117`), the file's own comment calling it "VOID, not evidence".
`--status` (near `scale/m2_units.py:398`) prints `len(done)`, which counts journal
lines rather than the intersection. Progress against the plan is **32/132**, and
every "N/132" in STATE.md and DONE.md that came from `--status` carries the same
off-by-one.

**2. THE SLOPES — VERIFIED, with a caveat on R-squared.** Recomputed from journal
k/n sums with the repo's own `scale/pivot_probe.py:66 loglog_slope` and an
independent OLS, agreeing to machine precision:

    pivot_signed__in_P   sizes [8,32,128,512]           slope +0.039581  R2 0.544949
                         sizes [8,32,128,512,1024]      slope +0.031436  R2 0.522986
                         sizes [8,32,128,512,1024,2048] slope +0.026990  R2 0.522208
    dense_signed__at_pivots
                         sizes [8,32,128,512,1024]      slope -0.746313  R2 0.829577
                         (s=2048 rate is 0.0 and is dropped, never clamped)

The prose numbers are **correct at three decimals for the size sets they used**:
DONE.md:3718 `+0.031` is the five-size fit; `-0.746` is the five-positive-size
fit. The prose `+0.031` predates the s=2048 pivot cell; with all six sizes the
pivot slope is **+0.027**, still far above the `-0.3` kill.

**The R-squared figures do not all reproduce from the journal.** DONE.md's pivot
values reproduce only from the six-decimal *display-rounded* rates, not from k/n:

    pivot 4 sizes:  exact 0.5449  |  display-rounded 0.5453  |  DONE.md says 0.5453
    pivot 5 sizes:  exact 0.5230  |  display-rounded 0.5228  |  DONE.md says 0.5228
    dense 5 sizes:  exact 0.8296  |  display-rounded 0.8295  |  DONE.md says 0.8296

So the dense R-squared was fitted on exact values and the pivot one on rounded
values. The slopes are unaffected at three decimals. Note also that the shipped
`loglog_slope` returns `(slope, npts)` and **computes no R-squared at all**
(`scale/pivot_probe.py:66-77`), so no R-squared in the M2 prose came from the M2
code path; the only such helper in `scale/` is `scale/lo_probe.py:45`.

**3. THE DETERMINISM CLAIM — the replay is real; "eight consecutive buckets" is
prose, not journal. VERIFIED by running one.**

The journal cannot carry it: `results/m2.jsonl` records `{key, meta{seconds},
value{k,n,rate,sigma,term}}` and nothing else — zero replay, determinism, match
or verification fields on any of the 33 lines. `scale/bucket.py:186` prints
`replay {key}: MATCH` to stdout and `run_bucket` returns it in `acc["verified"]`,
which `__main__` discards. It is never appended.

DONE.md holds **five** explicit records for this unit — 2591 ("again"), 2641
("third time"), 2693 ("fourth time"), 3743 ("eighth consecutive"), 3922 ("ninth
consecutive") — with the 1st, 5th, 6th and 7th not written down anywhere;
DONE.md:3543 separately says "replay-verified five times". Because
`run_bucket(..., verify=1)` walks `units()` in order and stops at the first
already-done key, every audit replays **the same single unit**,
`pivot_signed__in_P/s8`, at 4096 draws — not eight different cells.

`python scale/m2_units.py --status` gives `33/132 units journalled`, `next:
dense_signed__at_pivots/s2048/b2`, exit 0.

Replay run directly, without touching the journal or the lock (the loop holds
`results/m2.lock`, PID 2256):

    compute(pivot_signed__in_P/s8)  22.02 s CPU, CUDA_VISIBLE_DEVICES=""
    journal   {"k":101,"n":4096,"rate":0.024658203125,
               "sigma":0.03115874217982204,"term":0.006030712257871856}
    recompute {"k":101,"n":4096,"rate":0.024658203125,
               "sigma":0.03115874217982204,"term":0.006030712257871856}
    MATCH — bit-identical in all five fields.

**4. CALIBRATION — VERIFIED.** `python run_calib.py --self-test` gives **exit 0**.
The gate first rejected a deliberately-wrong target (`sgate hops=2` published
`0.164062501`, measured `0.1640625`, delta `-9.999999995e-10`, "G2 FIRES"), then
the real run:

    signed  hops=3   0.0468750000
    sgate   hops=1   0.0234375000
    sgate   hops=2   0.1640625000
    softmax hops=3   0.0000000000
    CALIBRATED: 4/4 bit-identical to the published values.

**5. THE FROZEN ITEM — VERIFIED, LOCK INTACT.** The M2 block in CHECKLIST.md
(from the `**M2. CONTEXT-STABLE SIGNED INFLUENCE AT GLOBAL REACH**` header
through the `Kill:` paragraph, CRLF to LF, whitespace-stripped, 592 bytes) hashes
to sha256 **`efadc390c93f`** — the value DONE.md and `scale/m2_units.py:3` claim.
`results/m2_item_text.txt` **exists** (601 bytes on disk, CRLF line endings) and
hashes to the same `efadc390c93f` under the same normalisation; a unified diff
against the CHECKLIST.md block is **zero lines**. The normalisation is
load-bearing: the raw bytes of `m2_item_text.txt` hash to `579e423a17a6`, and the
CHECKLIST.md slice with its trailing blank line hashes to `7851dfce9229`.

**6. LEAN — 27 theorems, zero `sorry`, build exit 0. VERIFIED.**
`lean/CEQ/*.lean` is five files. Lines matching `^\s*theorem\s` number 28
(Contraction 5, Nilpotent 4, Occupancy 3, OrbitBound 5, Refcount 11), but one of
those — `lean/CEQ/Refcount.lean:39`, `theorem over the ACTUAL surviving prefix
set, where the constraint is a consequence` — sits inside a `/- ... -/` block
comment and is English prose, not a declaration. **The actual theorem count is
27.** The docs' "27" is right; "12" is stale; a naive grep returns 28 and needs
that one manual exclusion.

`sorry`: **zero** in `lean/CEQ/*.lean`. The single token in the tree is
`lean/CEQ.lean:14`, prose in a doc comment reading "No `sorry` anywhere."
No `lemma`, `example`, or `axiom` declarations in any of the five files.

`cd lean && lake build CEQ` gives **exit 0**, no output (fully cached: 4100
mathlib oleans under `lean/.lake/packages/mathlib`, all five CEQ oleans present
and newer than their sources). Recorded for whoever reads the toolchain next:
`lean/lean-toolchain` pins `leanprover/lean4:v4.7.0` while `lake` on PATH reports
`Lake 5.0.0-src+819816b (Lean 4.33.1)`; elan has v4.7.0, v4.31.0 and v4.33.1
installed and resolves the pin, so the build is on v4.7.0.

**7. TEST COLLECTION — VERIFIED. 847.**
`CUDA_VISIBLE_DEVICES="" python -m pytest --collect-only -q tests/` gives
**`847 tests collected in 49.56s`**, **exit 0**. The suite itself was not run.
794, 809 and 829 are all stale.

**8. THE STRUCK CONSTANT — VERIFIED PRESENT. Live in shipped code and pinned by
two tests.**

    ceq/diagnose.py:91                       published_slope=-1.389, published_r2=0.9938,
    ceq/hf/modeling_ceq.py:133                                      "slope": -1.389, "r2": 0.9938,
    tests/cameron/test_diagnose_package.py:32   PUBLISHED_SLOPE = -1.389
    tests/chase/test_hub_package_hardening.py:497        128: 0.00391, "slope": -1.389, "r2": 0.9938,
    tests/chase/test_hub_package_hardening.py:531   assert "-1.389" in done or "1.389" in done

Docstring and prose hits: `ceq/diagnose.py:14`, `ceq/diagnose.py:63`,
`ceq/hf/modeling_ceq.py:25`, `ceq/hf/configuration_ceq.py:17`,
`tests/cameron/test_diagnose_package.py:12`. Also present in `scale/lo_probe.py`
(1, 5, 27, 116), `scale/pivot_probe.py:20`, `scale/tgate_probe.py:20`,
`scale/dfloor_probe.py:28`, `scale/sparse_probe.py:118`, and `README.md:865,901`.
DONE.md:3217 records the exponent as "**WITHDRAWN**"; the two shipped modules and
the two test pins still carry it.

**9. `pivot_unsigned` — the arm EXISTS; no unsigned pivot-routed measurement has
ever been journalled. VERIFIED.**

Exists at `scale/pivot_probe.py:106` (in the hand-maintained `ARMS` tuple),
`scale/pivot_probe.py:123` (dispatch: `pivot_unsigned` and `dense_unsigned` both
build `bench._softmax_operator(qq, kk)`, then `pivot_hop2` because the name
starts with `pivot`), and `scale/pivot_probe.py:228` (argparse default arm list).
Bound by `tests/loop/test_pivot_arms_distinct.py:11,92`.

Journalled: **no**. `results/m2.jsonl` is the only `.jsonl` journal in `results/`
and contains zero `pivot_unsigned` keys. `m2_units.CELLS`
(`scale/m2_units.py:80-107`) defines no `pivot_unsigned` cell — the closest is
`softmax_FLOOR`, which is `("softmax_FLOOR", "dense_unsigned", "not_in_P")`, i.e.
**not** pivot-routed, and it has 0 of 21 units run.

The only recorded run of the arm is the ad-hoc artifact
`results/iter03_floor_control.txt:6`, at `n=128` draws:

    pivot_unsigned    0.0000[0.000,0.028]   0.0000[0.000,0.028]  slope +nan (0/2 nonzero)

That file's own header states the run is void if an exact 0.0000 appears anywhere
until a known-truth synthetic passes.

**10. STAR-TRANSFORMER — VERIFIED by direct fetch, both sentences verbatim.**
arXiv:1902.09113, title **"Star-Transformer"**, Qipeng Guo, Xipeng Qiu, Pengfei
Liu, Yunfan Shao, Xiangyang Xue, Zheng Zhang, submitted **25 Feb 2019**. Fetched
via arxiv.org/abs and ar5iv. Both sentences quoted in CHECKLIST.md's G1 addendum
appear **verbatim** in Section 3.1:

  * "The relay node acts as a virtual hub to gather and scatter information from
    and to all the satellite nodes."
  * "With the radial connections, every two non-adjacent satellite nodes are
    two-hop neighbors and can receive non-local information with a two-step
    update."

CHECKLIST.md's addendum quotes each as a fragment of the sentence it appears in;
both fragments are exact.

### Not verified

* Which `s` values produced the two columns in `results/iter03_floor_control.txt`
  — the artifact does not state them and the probe's argparse default is three
  sizes, not two. UNVERIFIED.
* Whether the 1st, 5th, 6th and 7th replay audits asserted by the "eighth
  consecutive" count ever ran — no journal field, no DONE.md line, nothing on
  disk to read. Only the five recorded ones can be checked. UNVERIFIED.
* Whether `lake build CEQ` compiles from scratch: the run observed was a cached
  no-op (exit 0, no output). A cold build was not attempted, per the CPU rule.
  UNVERIFIED.


---

# CHASE — M2 PRE-GREEN AUDIT, iteration 23. 6 RED, 1 calibration GREEN.

`run_calib.py --self-test` exit 0, 4/4 bit-identical, gate observed rejecting a
wrong target first. Nothing in this audit touched the probe.

## The false green is not hypothetical, it is scheduled

`report()` and `report_decisive()` both REFUSE today (99/132 and 64/97 units
missing). They stop refusing the moment the bucket loop finishes. At that point
`pivot_signed__not_in_P` will be all-zero at s=32..2048, `loglog_slope` returns
`(nan, 0)`, `_verdict` maps NaN to `k2 = False`, and the run prints:

    slope(c NOT in P) = +nan -> control decays as the mechanism requires
    M2 = GREEN

RED: `tests/chase/test_m2_verdict_nan.py::test_an_unmeasured_control_must_not_produce_a_green`
RED: `tests/chase/test_m2_verdict_nan.py::test_verdict_rejects_nan_control_explicitly`
Site: `scale/m2_units.py::_verdict`, `k2 = (b > -0.3) if b == b else False`.

## `c not in P` is an identity, measured

s=32, k=8, seed 0, 8 draws: `not_in_P` moves the influence by max |delta| = 0.0,
8/8 draws bitwise identical; `A[i,j]` 0.0; `hop2[i,j]` 0.0. Same probe, `in_P`:
max |delta| = 0.20698779821395874 on 4/8 draws.
RED: `tests/chase/test_m2_not_in_P_is_structural_zero.py::test_a_not_in_P_draw_can_move_the_signed_influence`

## Four instrument defects the frozen text does not cover

1. **s=8 is one arm reported twice (G3).** `select_pivots` excludes i and j, so
   at s=8 |P| = 6 = every usable index and `A[:,P]A[P,:]` equals `A@A` at [i,j]
   to 1.86e-09 -- under the probe's own 1e-6 floor. `results/m2.jsonl` agrees to
   every digit: `pivot_signed__in_P/s8` and `dense_signed__at_pivots/s8` both
   k=101 n=4096 rate=0.024658203125. At s=32 the same delta is 1.3e-02.
   That shared point anchors BOTH log-log fits:

   | fit | with s=8 | without s=8 |
   |---|---|---|
   | pivot | +0.0270 | +0.0044 |
   | dense | -0.7463 | **-1.0093** |

   The separation is real and gets *stronger* without s=8; the published slopes
   are the ones that are wrong.
   RED: `test_m2_instrument_binds.py::test_the_pivot_and_dense_arms_are_distinct_at_s8`

2. **|P| = 6 at s=8, not the k=8 every artifact states.**
   RED: `test_m2_instrument_binds.py::test_k_is_the_reported_8_at_every_size_in_the_table`

3. **The M2 operator ships nowhere.** `pivot_signed` builds A with
   `_causal_tgate_operator`, `g_i*tanh(qhat.khat/tau)`, denominator-free, which
   `grep` finds in `ceq/bench.py` and nowhere else. The Hub ships `sgate`,
   `rho*(softmax(w) - lam*softmax(-w))/(1+lam)`; `ceq/attention.py` ships
   `rho*w/||w||_1`. Both shipped forms carry the denominator `ceq/bench.py:242`
   itself names as "the measured cause of the 1/s death".
   RED: `test_m2_instrument_binds.py::test_the_m2_signed_arm_uses_an_operator_that_ships`

4. **No M2 arm is windowed.** `build_arm` never passes `window=`. The `windowed`
   label picks the pool `c` is drawn from, `range(i-8, i)`, while hop 2 stays
   routed through the same global P. That cell measures the chance a window
   token happened to be a pivot: 88/320 at s=32, 18/320 at s=128, 3/320 at
   s=512 -- it decays as k/s and says nothing about reach.
   RED: `test_m2_instrument_binds.py::test_the_windowed_arm_actually_windows_the_operator`

## Completeness, 32/132 units

`softmax_FLOOR` 0/6 cells. `random_CEILING` 0/6 cells. `dense_signed` 0/6.
`pivot_signed__windowed` 0/6. `dense_signed__at_pivots/s2048` 2/8 batches.
`m2_units.py` calls the floor and ceiling "instrument ends: without these
neither slope means anything" -- neither has a single draw.

## The draw-count defense is false

`scale/m2_units.py:52` and `STATE.md:101,118`: "The claim arm keeps its full
pre-registered 16384 at every size". The journal has **4096** at s=8/32/128/512
and 16384 only at s=1024/2048. `results/m2_run.txt` states this correctly. The
frozen text says "16,384 draws at the tail", so the measurement is compliant;
the sentence defending the control's reduction is not.

`clopper_pearson(0, 4096) = 0.00090` is correct arithmetic that describes no
dense cell. The only dense zero is s=2048 at n=1024, CP hi **0.00360**.
0.00090 is exactly the `not_in_P` cells' bound.

## Blast radius

No M2 number has reached a shipping file. The struck `-1.389` has:
`ceq/hf/modeling_ceq.py:133` `COSTS["content_conditional_sign_decay"]["slope"]`
and `configuration_ceq.py:17` -- both uploaded by `ceq/hf/train.py:279`
`upload_folder`. `ceq/diagnose.py:91` emits `published_slope=-1.389` into
`--json`. The withdrawal lives only in `README.md:79` and `MODEL_CARD.md:95`,
neither of which ships. Two tests PIN the struck value:
`tests/cameron/test_diagnose_package.py:32,68` and
`tests/chase/test_hub_package_hardening.py:496-497,531`.


---

## LOOP ITERATION 24 — 2026-08-25 — WILSON CORRECTED FOUR OF MY CLAIMS. Two were instrument defects; both fixed.

Wilson recomputed everything from `results/m2.jsonl` rather than from any table I
wrote. His facts settle every dispute below. **Four corrections land against me.**

**CORRECTION 1 — every `N/132` I published was ONE HIGH.** `--status` printed
`len(done)`, i.e. journal LINES, not members of `units()`. The void
`pivot_signed__not_in_P/s8` unit (rate 0.0 from **n=0** draws) is journalled but
excluded by `IMPOSSIBLE`, so it inflated every count. **Real progress is 33/132,
not 34.** The wrong number was copied into STATE.md and DONE.md across several
iterations. FIXED [RUN]: `--status` now counts the intersection and NAMES the
orphan — `33/132` plus `(1 journalled but NOT in the plan:
['pivot_signed__not_in_P/s8'])`.

**CORRECTION 2 — the replay audit was ONE unit N times, not N units.**
`verify=1` walked `units()` and stopped at the first done key, so every bucket
replayed `pivot_signed__in_P/s8` and nothing else. "Replay verified for the ninth
consecutive bucket" was **true and misleading**: one cell nine times. A drift in
any other cell would never have been caught. FIXED [RUN]: the target now rotates
by `len(done) % len(candidates)`, verified to walk s8 -> s32 -> s128 -> s512 ->
s1024 across successive buckets.

**CORRECTION 3 — my R² values came from DISPLAY-ROUNDED rates.** DONE.md's pivot
R² 0.5453 and 0.5228 reproduce only from the rounded table values; exact journal
values give 0.5449 and 0.5230. Worse, and this is the real finding: **the shipped
`loglog_slope` returns `(slope, npts)` and computes no R² at all**, so NO R² in
any M2 prose came from the M2 code path — I computed them in recording scripts
from numbers I had already rounded for a table. The SLOPES are correct at three
decimals (+0.031436 and -0.746313, confirmed by two independent fits); the R²
figures are not evidence and should not be quoted as though they were.

**CORRECTION 4 — test collection is 847.** Not 794, not 809, not 829. All three
appear in current docs and all three are stale. Suite still never run.

**WILSON'S VERIFICATIONS THAT HOLD:**
  * LOCK **efadc390c93f** intact — CHECKLIST slice and `results/m2_item_text.txt`
    hash identically under the same normalisation (which is load-bearing: raw
    bytes give 579e423a17a6, and including the trailing blank line gives
    7851dfce9229).
  * Lean **27 theorems** (naive grep says 28; `Refcount.lean:39` is English prose
    inside a block comment). Zero `sorry`. `lake build CEQ` exit 0 — **but cached**,
    so a cold build is UNVERIFIED.
  * Calibration exit 0, gate rejected a 1e-9-off target first, 4/4 bit-identical.
  * Star-Transformer 1902.09113 — both G1 sentences appear **VERBATIM** in
    Section 3.1. The near-miss is real and correctly quoted.
  * He ran the replay himself: 22.02 s, **bit-identical in all five fields**.

**TWO THINGS WILSON FOUND THAT CHANGE WHAT COMES NEXT:**

  1. **The struck `-1.389` is LIVE IN SHIPPED CODE**, not merely in prose:
     `ceq/diagnose.py:91` (`published_slope=-1.389`) and
     `ceq/hf/modeling_ceq.py:133` (`"slope": -1.389`), **pinned by three tests**
     (`tests/cameron/test_diagnose_package.py:32`,
     `tests/chase/test_hub_package_hardening.py:497` and `:531`). DONE.md:3217
     records it WITHDRAWN. A withdrawn number that ships inside the package is
     the same class of defect as the 84x — and it would go to HuggingFace.

  2. **`pivot_unsigned` EXISTS as an arm and has NEVER been journalled.**
     Defined at `scale/pivot_probe.py:106,123,228`, bound by
     `tests/loop/test_pivot_arms_distinct.py`. Zero keys in the journal. And
     `softmax_FLOOR` is `("softmax_FLOOR", "dense_unsigned", "not_in_P")` — **NOT
     pivot-routed**, so it is not the S2 comparator either. The one recorded run
     is ad hoc at n=128 reading `0.0000 [0.000, 0.028]` with slope `+nan
     (0/2 nonzero)` — and which `s` values produced those two columns is
     UNVERIFIED. **S2 genuinely has no data.** Cameron is on it.

CHECKLIST: no status changed. M2 **33/132** (corrected), UNTESTED.

## The same defect, elsewhere in CHECKLIST.md

**M4 is M2's defect on a tensor.** Kill: "max change > 1e-12 in the
evict-before-read window". `ceq/eviction.py:71-76` picks the perturbed token via
`lowest_salience_token(..., exclude=keep)`, and `settle_evicted`
(`ceq/eviction.py:118-121`) reads `x[keep]` and nothing else. The perturbed
token can never be in `keep`, so the change is 0.0 by construction. Measured:
perturbed token inside `keep` in **0 of 8** draws, max change exactly 0.0 in
**8 of 8**. `tests/w3/test_w3_eviction.py:96` asserts `e == 0.0` -- an identity.
`evicted_operator`'s own docstring states the tautology outright: "perturbing it
is bitwise invisible". The companion assertion `g > 0.0` on the gated arm IS a
real measurement; only the eviction half is empty.
RED: `tests/chase/test_checklist_kills_are_evaluable.py::test_m4_max_change_under_eviction_is_capable_of_exceeding_1e_minus_12`

**M5 clause is half-bound.** `StrictlyLower` is checked numerically on real
tensors. The truncation index is not: the theorem is stated at `N = n`
(`lean/CEQ/Occupancy.lean:42`), the module truncates at `DEFAULT_HOPS = 4`
(`ceq/attention.py:88`) and ships at `hops = 2`
(`ceq/hf/modeling_ceq.py:329`), where `A^N != 0`. The grep-binding
`tests/w3b/test_w3b_lean_nilpotent.py:105-118` contains the substring "hops"
zero times.
RED: `...::test_m5_the_lean_truncation_index_hypothesis_is_bound_to_the_shipped_hops`

**M3, M6, S3, G4 name quantities nothing measures.**
`scale/negation_scope.bootstrap_ci` has zero callers; no softmax arm exists.
No module computes a numerical radius, so M6's guard does not exist
(`ARSENAL.md:71` records `w(A) = 1.499315` and calls the bound vacuous). No file
implements the Carnot grading. "G4" appears in one Python docstring and in no
evaluator -- G2 has `run_calib.py`, G3 has `tests/loop/test_arms_distinct.py`,
G4 has nothing.

## The determinism audit has audited one unit, nine times

`scale/bucket.py::run_bucket` with `verify=1` replays the FIRST journalled key
in `units()` order, which is always `pivot_signed__in_P/s8` -- 4.24 s, and the
degenerate cell (|P| = 6, arms identical). All 16 s=2048 batches are journalled;
none has ever been replayed. `STATE.md`'s "Replay verified for the ninth
consecutive bucket" is nine replays of the same four seconds.


---

## LOOP ITERATION 26 — 2026-08-25 — dense control at s=2048: 0 of 2048 draws, 4 of 8 batches.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
Replay MATCH -- eleventh consecutive determinism audit.

PID 2256 finished and released the lock cleanly, having journalled b2. This
bucket then ran b3 (377 s, over the 250 s budget because the check happens
before a unit starts, not during -- the unit was allowed to complete rather than
be truncated, which is the intended behaviour).

**[RUN] dense_signed__at_pivots/s2048, 4 of 8 batches: k=0, n=2048.**

Four independent batches, four zeros. Clopper-Pearson 95% upper bound on 0/2048
is **0.001462**, against the claim arm's **0.029907** at the same s:

    separation at s=2048  >=  **20.5x**   (a BOUND, not an estimate)

Running separation, pivot / dense:
    s=8 1.00x | s=32 1.3x | s=128 3.0x | s=512 9.1x | s=1024 61x
    s=2048 **>= 20.5x** (bound, 4 of 8 batches)

The bound has tightened 5.2x -> 10.3x -> 20.5x as batches land, exactly as an
all-zero cell should: the point estimate cannot move below zero, so every
additional batch buys only interval.

**Still not a verdict.** 4 batches remain; `--decisive` requires the cell
complete. The pre-registered reading is fixed in `M2_PREREGISTERED_READING.md`
and says outcome A is a MECHANISM result -- not capability, not novelty, and
attributable to ROUTING rather than signedness until S2's pivots+unsigned
ablation runs.

CHECKLIST: no status changed. M2 is 35/132 and remains UNTESTED.

### S2 — THE ABLATION, RUN (Cameron). Kills evaluated against the table above, unmoved.

**Correction to the dispatch's premise, first, because it was the reason for the
run.** The dispatch said `pivot_unsigned` "HAS NEVER BEEN RUN". Not quite: it ran
once, at iteration 3, at `n=128` draws and two sizes, as a floor control —
`DONE.md:1810-1817` and `results/iter03_floor_control.txt:5-6`, both cells
`0.0000[0.000,0.028]`. What is true is that it has never entered
`results/m2.jsonl` (`grep -ciE "unsigned|FLOOR|CEILING" results/m2.jsonl` = 0),
never run at M2's protocol scale, and never been read as an S2 ablation. The
conclusion the dispatch drew stands; its premise needed narrowing.

**Instrument.** `scale/s2_probe.py`, new, standalone — it drives
`pivot_probe.run_arm` and `pivot_probe.build_arm` and swaps exactly one symbol
per arm. `scale/pivot_probe.py` (mtime 10:53:09, before this work), `m2_units.py`
and `results/m2.jsonl` were not written. Bind: `tests/cameron/test_s2_ablation.py`,
RED first (`ImportError: cannot import name 's2_probe'`, 4 tests uncollectable),
then GREEN 5/5. The absmag bind was RED a second time at
`-0.5365633964538574` — `min_influence_entry` held a `from ... import build_arm`
name bound at import, so it scanned the SIGNED operator and would have published
that number under the unsigned arm's name. That is the G3/ParaFormer bug class,
caught by the bind rather than by review.

`[RUN] python run_calib.py --self-test` -> **exit 0**, gate observed rejecting a
wrong target first, then 4/4 bit-identical (0.046875 / 0.0234375 / 0.1640625 /
0.0). **G2 does not fire.**

#### The pre-registered arms, as pre-registered

`n_draws=1024` (pivot) / `512` (dense), k=8, PROTOCOL: SCALING, c in P, seed 0.

| arm | s=8 | s=32 | s=128 | s=512 | slope(rate) | slope(R) |
|---|---|---|---|---|---|---|
| `pivot_unsigned` rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — (0/4 nonzero) | |
| CP95 | [0,0.00360] | [0,0.00360] | [0,0.00360] | [0,0.00360] | | |
| `pivot_unsigned` R=term/sigma | 0.067931 | 0.015919 | 0.000002 | 0.000000 | | **-5.471** |
| `dense_unsigned` rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — (0/4 nonzero) | |
| CP95 | [0,0.00718] | [0,0.00718] | [0,0.00718] | [0,0.00718] | | |
| `dense_unsigned` R | 0.056340 | 0.000670 | 0.000001 | 0.000000 | | **-6.282** |
| `randpivot_signed` rate | 0.023438 | 0.026367 | 0.028320 | 0.033203 | **+0.081** | |
| CP95 | [0.01507,0.03467] | [0.01745,0.03813] | [0.01905,0.04042] | [0.02310,0.04609] | | |
| `randpivot_signed` R | 0.174011 | 0.170490 | 0.159748 | 0.155975 | | -0.028 |

Journalled arms, DERIVED from `results/m2.jsonl` as read at 11:40 (33 lines,
6007 bytes), n-weighted across batches — `term` and `sigma` have been written to
that journal since M2 began and had never been read back by anything:

| arm | R at s=8 | 32 | 128 | 512 | 1024 | 2048 | slope(rate) | slope(R) |
|---|---|---|---|---|---|---|---|---|
| `pivot_signed__in_P` | 0.193548 | 0.160129 | 0.160447 | 0.153208 | 0.151240 | 0.151551 | +0.027 | **-0.038** |
| `dense_signed__at_pivots` | 0.193548 | 0.063022 | 0.018662 | 0.005009 | 0.002418 | 0.001266 | -0.746 | **-0.913** |

#### The pre-registered comparator is CONFOUNDED, and the confound is measured

`bench._softmax_operator` scores `q.k/sqrt(d)` on UNNORMALIZED q,k; `tgate`
scores `qhat.khat/tau` with both L2-normalized. Measured, 24 draws, row i=s-1:

| s | softmax logit std | max row weight | participation ratio 1/sum(a^2) | tgate mean abs entry |
|---|---|---|---|---|
| 8 | 14.412 | 0.939805 | 1.145 | 0.113393 |
| 32 | 15.609 | 0.899966 | 1.290 | 0.100202 |
| 128 | 15.769 | 0.887753 | 1.269 | 0.101982 |
| 512 | 15.775 | 0.835463 | 1.505 | 0.104021 |

A participation ratio of 1.15-1.51 is a hard argmax, not a mixture. That arm's R
measures "did the argmax land on a pivot", which falls like k/s for reasons that
have nothing to do with signedness. **The proof that this is fatal and not
cosmetic is that the softmax comparator does not separate the axis it was
supposed to separate**: routed -5.471 against dense -6.282, a difference of 0.81
in the exponent, where the same contrast on a magnitude-matched operator is
-0.046 against -0.930, a difference of 0.88 — same size, opposite verdict about
whether routing does anything.

#### The corrected comparator: `absmag` = |tgate|

The same matrix with the sign stripped. Same scores, same tau, same per-row gate,
identical magnitudes entrywise, no denominator; the only thing that changes is
whether the background sum can CANCEL. Bound by requiring c's own `term` to come
out bit-identical to the signed arm. **Reported as POST-HOC, and it does not
replace the pre-registered number above.**

| arm | R at s=8 | 32 | 128 | 512 | rate | slope(R) |
|---|---|---|---|---|---|---|
| `pivot_absmag` | 0.187266 | 0.185428 | 0.170926 | 0.155636 | 0.0, CP95 [0,0.00360] | **-0.046** |
| `dense_absmag` | 0.183236 | 0.067886 | 0.015344 | 0.004095 | 0.0, CP95 [0,0.00718] | **-0.930** |

#### THE 2x2, on the two metrics, magnitudes matched

|  | slope(rate) — only a signed operator can score | slope(R) — both can score |
|---|---|---|
| **pivot, signed** | **+0.027** flat | **-0.038** flat |
| **pivot, unsigned** | 0.0 at every s, by theorem | **-0.046** flat |
| **dense, signed** | -0.746 decays | -0.913 decays |
| **dense, unsigned** | 0.0 at every s, by theorem | -0.930 decays |

**Read it down the columns.** `R` separates cleanly on the ROUTING axis
(-0.04 against -0.92) and is INDIFFERENT to the signedness axis (-0.038 vs
-0.046 routed; -0.913 vs -0.930 dense — the signed and unsigned members of each
pair agree to within 0.008 and 0.017 in the exponent). `rate` is at an exact
structural floor for both unsigned arms at every size.

#### Verdicts against the frozen kills

| kill | fires? | number |
|---|---|---|
| **K1** | **NO** | requires rate > 0; measured 0/1024 at all four sizes, CP95 [0, 0.00360] |
| **K2** | **YES** | `pivot_unsigned` AND `dense_unsigned` exactly 0.0 at every s. **K1 is UNEVALUABLE** — the signedness axis is a structural floor on `rate`, exactly as pre-registered |
| **K3a** | as written NO (pivot_unsigned -5.471 < -0.3); on the corrected comparator its CONDITION IS MET, -0.038 and -0.046, both > -0.3 | routing holds dilution-resistance flat **independently of signedness** |
| **K3b** | **NO** | pivot_signed slope(R) = -0.038 > -0.3. The small-ball/share explanation of the flat rate SURVIVES |
| **K3c** | **NO** | dense_unsigned -6.282, dense_absmag -0.930, both < -0.3 |
| **K4** | **YES** | `randpivot_signed` slope(rate) = **+0.081** > -0.3 |

**G4 DOES NOT FIRE.** G4 reads "S2 shows the capability lives entirely in the
unsigned ablation". The capability metric is 0/1024 and 0/512 in the unsigned
ablation at every size, routed and dense alike, with CP95 upper bounds 0.00360
and 0.00718 against the routed signed arm's 0.0247-0.0332. It does not live
there at all. What DOES live there is the flatness — and the flatness is not the
capability.

**K4 FIRES, and this is the real cost of the run.** `randpivot_signed` uses k=8
pivots drawn UNIFORMLY, reading nothing of the content, and is flat at +0.081
with every CP interval overlapping the content-selected arm's. Content selection
is **not load-bearing for M2**. Per the pre-registered consequence, the words
"content-selected" must be struck from the M2 sentence: what M2 measures is
FIXED PATH COUNT, which is precisely the axis Star-Transformer 1902.09113 (2019)
occupies with a content-blind virtual hub.

**Why K4 firing was predictable from a document already in this repo, and why
that makes it stronger rather than weaker.** `scale/recall_probe.py:3-8` states
the identity: for any CONTENT-BLIND schedule of size k,
`P(c reachable) * (share | reachable) = (k/s) * (1/k) = 1/s`, bit-for-bit the
dense rate. M2 draws c FROM P, so it measures the second factor and conditions
the first away. A content-blind schedule must therefore come back flat on M2 —
and it did, at +0.081. That file was written, never run, and its conclusion never
reached the M2 design. The measurement above is the first time the identity has
been checked against a number.

**The consequence, stated as a limit on the instrument and not on the result:
M2 CANNOT measure content selection, for any operator, by construction.** The
separating quantity is the UNCONDITIONAL rate, `rate * P(c in P)`, and that
requires a c designated from OUTSIDE the selector — a token the task says
matters. M2 has no such token; it draws c from whatever landed in P. M3
(negation-scope, ground-truth token at distance d) does have one. So the
content-selection half of the claim is an M3 question that was being asked of an
M2 instrument, and no number of additional M2 draws can answer it.

#### What the ablation leaves the claim sentence able to say

Not "either signedness wins or Star-Transformer owns it". The two axes are
orthogonal and the table measures them separately:

* **Fixed path count** buys dilution-resistance, `slope(R)` -0.04 against dense
  -0.92, and buys it with or without signs (-0.038 signed, -0.046 unsigned). That
  is Star-Transformer's axis, published 2019, and the claim must credit it.
* **Content selection** buys nothing measurable on M2 (+0.081 for random pivots).
  Unresolved, not refuted — M2 is the wrong instrument for it.
* **Signedness** is the only axis on which the unsigned construction sits at an
  exact structural floor at every size. It is what makes the flat quantity a
  SIGN rather than a share.

Neither factor is sufficient: unsigned+routed is flat and cannot represent a
sign flip at all; signed+dense can represent one and loses it at -0.746. The
novel cell is the CONJUNCTION, and the ablation is what shows the conjunction is
not decomposable into either half.


---

## LOOP ITERATION 27 — 2026-08-25 — S2 built and ready. The ablation that can overturn M2.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**LOCK HELD** by PID 34856, measured COMPUTING (23.9 CPU-s over 12 s wall). No
second copy started. Non-competing action taken: build S2 without running it.

ACTION (one): wrote `scale/s2_units.py` -- the pivots+unsigned ablation, on its
own journal so it can never block M2's `--report`.

**WHY THIS IS THE NUMBER THAT MATTERS.** M2 shows ROUTING does something: the
claim arm is flat (slope +0.0270) while a dense control at the SAME intervened
tokens decays (61x apart at s=1024). It does NOT show that SIGNEDNESS does.
**Star-Transformer (arXiv:1902.09113, 2019) already routes hop-2 through a relay
with an UNSIGNED operator.** If unsigned routing is also flat, the contribution
is routing, it belongs to 2019, and stopping condition **G4** fires.

**THE THREE OUTCOMES, FIXED IN THE FILE BEFORE IT RUNS:**
  * unsigned routed FLAT -> **G4 fires**, contribution is routing, rewrite the
    claim sentence before publishing anything.
  * unsigned routed DECAYS while signed stays flat -> signedness is load-bearing
    ON TOP of routing; that combination is the unoccupied cell.
  * unsigned routed at a STRUCTURAL ZERO -> **the comparison is VACUOUS and must
    not be read as a win.** Softmax has `I + A + A^2` non-negative entrywise and
    so cannot flip a sign at all. **This is the likely outcome** -- and the
    report function prints that verdict itself rather than leaving it to
    whoever reads the table. If it happens, S2 needs an unsigned arm that CAN
    flip: softmax with a signed value path, or the `wrt="x"` channel where
    softmax does have a sign to lose.

That third branch is the one worth having written down. The obvious S2 arm is
`pivot_unsigned`, and the obvious reading of "unsigned reads 0.0 everywhere"
is "signedness wins" -- which would be **exactly wrong**, and exactly the shape
of the multizoom and dfloor artifacts: a number that is what it is by
construction, dressed as a result. The file now refuses that reading in code.

CHECKLIST: no status changed. M2 is 35/132; S2 is 0/21 and not yet started.


---

## LOOP ITERATIONS 25-26 — 2026-08-25 — CHASE: 13 RED. M2 MUST NOT GO GREEN. The false green was ON A TIMER; fixed.

CALIBRATION [RUN] exit 0, 4/4 bit-identical. Bucket 10 -> **35/132** (corrected
count; `dense_signed__at_pivots/s2048/b4` = 0/512, 402 s).

**THE FINDING THAT CHANGED WHAT I DID NEXT.** `_verdict` contained

    k2 = (b > -0.3) if b == b else False

mapping the NaN slope of an all-zero series to False, i.e. to "control decays as
the mechanism requires" -- and GREEN follows. Chase confirmed it prints, verbatim:

    slope(c NOT in P) = +nan -> control decays as the mechanism requires
    M2 = GREEN

`report()` refuses TODAY only because units are missing. **It stops refusing the
moment the bucket loop finishes, and the NaN fires exactly then.** The false green
was scheduled, and I was the one running the schedule -- every bucket I ran moved
it closer. **Fixed [RUN]:** NaN or a fit with fewer than two points is now
UNEVALUABLE, and any unevaluable mandatory clause forces **VOID**, never GREEN.
Verified on the exact scenario: it now prints "M2 = VOID -- a mandatory clause is
UNEVALUABLE", while a genuinely evaluable pair still returns GREEN.

**CHASE'S 13 RED, the ones that change the record:**

**(a) `c NOT in P` is an IDENTITY, measured.** s=32, k=8, seed 0: `max|delta
influence| = 0.0`, **8/8 draws bitwise identical**; against **0.20698779821** on
4/8 for `in_P`. Not a small number -- exactly zero.

**(b) M4 HAS THE SAME DEFECT.** `lowest_salience_token(..., exclude=keep)`
(`ceq/eviction.py:71-76`) guarantees the perturbed token is OUTSIDE `keep`, and
`settle_evicted` reads `x[keep]` and nothing else. Measured: perturbed token
inside `keep` in **0/8** draws, max change exactly 0.0 in **8/8**.
`tests/w3/test_w3_eviction.py:96` asserts `e == 0.0` -- **a tautology**. The
docstring says it outright: "perturbing it is bitwise invisible." **Two mandatory
items now have kills that cannot fire.** Same root pattern; fix the pattern.

**(c) THE s=8 COLUMN IS ONE ARM REPORTED TWICE -- G3.** `select_pivots` excludes
i and j, so at s=8 only 6 indices remain and **P is all of them** (|P| = 6, not
the k=8 every artifact states). The dropped p=i, p=j terms vanish on a strictly
lower triangular A, so `A[:,P]A[P,:]` equals `A@A` at [i,j] to **1.86e-09** --
under the probe's own 1e-6 floor. The journal agrees to every digit: both s=8
cells are `k=101 n=4096 rate=0.024658203125`. **And that shared point is the
leftmost point of BOTH fits.**

    fit      with s=8    without s=8
    pivot    +0.0270     +0.0044
    dense    -0.7463     **-1.0093**

**The separation is real and gets STRONGER without s=8. My published -0.746 is
the wrong number.** Three dense slopes are now in circulation (-0.746, -0.792,
-1.009); **-1.009 is the correct one** and the others must be withdrawn.

**(d) THE M2 OPERATOR SHIPS NOWHERE -- the sixteenth instrument.**
`pivot_signed`/`dense_signed` build A with `_causal_tgate_operator`,
`g_i * tanh(qhat.khat/tau)`, which exists in `ceq/bench.py` and **no other file**.
The Hub ships `sgate`; `ceq/attention.py` ships `rho*w/||w||_1`. **Both shipped
forms carry the denominator that `ceq/bench.py:242` itself names as "the measured
cause of the 1/s death".** M2 is textually compliant with "bench.py pivot arm",
but the CLAIM SENTENCE and the model card describe a module that was never
measured for M2.

**(e) MY OWN SENTENCE WAS FALSE.** `m2_units.py:52` and STATE.md say "the claim
arm keeps its full pre-registered 16384 at every size". The journal has **4096**
at s=8/32/128/512 and 16384 only at s=1024/2048. The frozen text promises "16,384
draws at the tail", so the MEASUREMENT is compliant -- the sentence defending the
control's reduction is not. I repeated it in the brief to Chase.

**(f) THE INSTRUMENT ENDS HAVE ZERO DRAWS.** `softmax_FLOOR` **0/6 cells**,
`random_CEILING` **0/6**. `m2_units.py` calls them "instrument ends: without these
neither slope means anything."

**(g) `pivot_signed__windowed` MEASURES THE WRONG THING.** `build_arm` never
passes `window=`; the label only picks the pool c is drawn from. It measures the
chance a window token happened to be a pivot (88/320, 18/320, 3/320 at
s=32/128/512) and decays as k/s. The frozen Test field's third arm is not testing
reach.

**(h) THE -1.389 STILL SHIPS, AND TWO TESTS FENCE OUT THE CORRECTION.**
`ceq/hf/modeling_ceq.py:133` (in `COSTS`, **uploaded**),
`ceq/hf/configuration_ceq.py:17` (**uploaded**), `ceq/diagnose.py:91`,
`ceq/hf/smoke.py:290`. The withdrawal lives only in README/MODEL_CARD, **neither
of which ships**. `tests/cameron/test_diagnose_package.py:32,68` and
`tests/chase/test_hub_package_hardening.py:496-497,531` assert the struck value,
so correcting it turns them RED. **A fresh wrong M2 number would land
identically** -- number -> COSTS -> guard test pins it -> retraction goes to a
file that does not travel.

**GOOD NEWS, and it is real:** **no M2 number has reached a shipping file.** Zero
hits in `ceq/`, `colab/`, no COSTS entry, no test assertion.

**M2 STATUS: still UNTESTED, and it stays that way.** The kill has not fired; the
INSTRUMENT is convicted. Per the loop law a RED is overturned only by convicting
the instrument -- here the instrument was convicted before any verdict, which is
the order that saves the result rather than the one that fakes it.

**FIVE ENGINEERING NURSES dispatched under Chase** on the user's instruction:
measurement throughput (535 ms/draw), the suite that never finishes (9 attempts,
847 collected), training checkpoint/resume (the HF blocker), kernel/memory
layout, and packaging hygiene (**no requirements.txt, setup.py or pyproject.toml
exists anywhere**). **TWO NURSES under Cameron** on free-compute sizing.


---

## LOOP ITERATION 28 — 2026-08-25 — S2 came out VACUOUS, exactly as pre-registered.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
M2 lock still held by PID 34856; S2 has its own lock, so this iteration did real
measurement instead of waiting.

**[RUN] S2 bucket 1 -- `pivot_unsigned__at_pivots`:**

    s=   8   rate 0.0   k=0  n=4096   (22s)
    s=  32   rate 0.0   k=0  n=4096   (22s)
    s= 128   rate 0.0   k=0  n=4096   (49s)
    s= 512   rate 0.0   k=0  n=4096  (209s)

**EXACTLY ZERO AT EVERY SIZE. This is outcome three, and it is VACUOUS, not a
win.** A non-negative operator has `I + A + A^2` non-negative entrywise, so it
cannot flip a sign at all -- with or without routing. The measurement says
NOTHING about whether routing or signedness carries M2's flatness.

**THE PRE-REGISTRATION EARNED ITSELF IN ONE ITERATION.** Written yesterday, before
this ran:

  > unsigned routed at a STRUCTURAL ZERO -> the comparison is VACUOUS and must
  > not be read as a win. **This is the likely outcome.**

Without that, the natural reading of "unsigned reads 0.0 everywhere while signed
reads 0.029" is *"signedness wins"* -- which is **exactly backwards**, and is the
same shape as the multizoom `c = s//2` artifact and the dfloor `min(4, nblk)`
ceiling: a number that is what it is by construction, dressed as a result.
`scale/s2_units.py::report` prints the refusal in code rather than trusting a
reader to notice.

**S2 MUST BE REDESIGNED, and the file already names the fix.** An unsigned arm
that CAN flip is needed. Two candidates, both recorded before the run:
  1. softmax with a SIGNED VALUE PATH;
  2. the **`wrt="x"` channel**, where softmax DOES have a sign to lose -- the
     repo's own record has softmax on `wrt="x"` reading 0.025391 / 0.009766 /
     0.000000, so it flips there and decays.

(2) is the cleaner comparison and is the next action. Blocker: `run_arm` in
`scale/pivot_probe.py` hardcodes the gradient target to `v` and has no `wrt`
parameter, so this needs a small, RED-first extension.

**Note what S2 did NOT do:** it did not weaken M2. M2's claim arm and dense
control are both measured `wrt="v"`, both can flip there, and their 61x
separation at s=1024 stands. What is unresolved is the ATTRIBUTION -- routing
versus signedness -- and it stays unresolved, which is the honest state.

CHECKLIST: no status changed. M2 35/132, S2 4/21 and its arm needs replacing.


---

## LOOP ITERATION 29 — 2026-08-25 — S2 rebuilt on a channel where the ablation can actually decide.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

ACTION (one): added a `wrt` channel to `run_arm` and rebuilt S2 on it, RED-first.

**THE FIX.** S2's ablation was vacuous because it ran on `wrt="v"`, where a
non-negative operator is pinned at exactly zero by theorem -- `I + A + A^2` is
non-negative entrywise, so softmax cannot flip a sign there AT ALL, routed or
not. `wrt="x"` is the representation path, where `A` itself depends on `x`, so a
softmax arm DOES have a sign to lose.

Implementation note, from bench.py's own hard-won pattern: the perturbation must
be assigned BEFORE `requires_grad_`, because an in-place write to a leaf raises.

**[RUN] RED-FIRST, s=64, 512 draws, in_P -- the precondition for S2 meaning
anything:**

    pivot_unsigned  wrt=v  0.000000  (0/512)   <- structurally pinned
    pivot_unsigned  wrt=x  **0.003906**  (2/512)   <- CAN flip
    pivot_signed    wrt=v  0.013672  (7/512)
    pivot_signed    wrt=x  0.041016  (21/512)

**The unsigned arm moves on `x`. So a zero there would now be a RESULT rather
than a tautology** -- which is the entire precondition, and it is why this had to
be proved before any S2 number was taken rather than after.

The four vacuous `wrt="v"` units stay in the journal as evidence of the
structural zero. They are not deleted and they are not the ablation.

**[RUN] S2 bucket 2 -- `pivot_signed__x`, the claim arm on the new channel:**

    s=   8  0.026367  k=108/4096   (35s)
    s=  32  0.034912  k=143/4096   (35s)
    s= 128  0.024902  k=102/4096   (57s)
    s= 512  0.031250  k=128/4096  (194s)

Flat, consistent with the `wrt="v"` claim arm (0.024658 / 0.028564 / 0.031006 /
0.028809). **The property does not depend on which channel it is measured in** --
which is itself a small robustness result the project did not have before.

The number that decides G4 is `pivot_unsigned__x`, and it has not run yet.

CHECKLIST: no status changed. M2 35/132; S2 8/42.

---

# CHASE — SUCCESSORS. Every disproof above, with what stands in its place.

Rule: a RED still stops build work on its item. What follows is the replacement
each dead clause owes, why it survives the objection that killed the original,
and its pre-registered kill.

## The general successor, which subsumes four of the six

**LIVENESS PRECONDITION ON EVERY KILL CLAUSE.** Before a clause's number may be
read, the measuring code must demonstrate on a known-positive that the quantity
CAN take a killing value. A clause whose detector has never been observed firing
is UNEVALUABLE, not PASSED.

Why it survives the objection: it is exactly the defect class, stated as a
precondition rather than caught case by case. M2 clause 2 (`c not in P` is an
identity), M4 (`max change` is an identity) and the NaN-to-PASS mapping all fail
it, and no new measurement is needed to see that they do.
Not a new abstraction: `tests/cameron/test_s2_ablation.py::test_the_negativity_detector_is_calibrated`
already is this pattern. It is being generalised, not invented.
**Pre-registered kill for the rule itself:** if a clause passes its liveness
check and still cannot be made to fire on any input, the clause is a theorem and
must be moved out of the kill list into the baseline-impossibility line, where a
proof belongs.

## M2 clause 2 -- `c NOT in P is ALSO flat`

DEAD: `hop2[i,j] = sum_{p in P} A[i,p] A[p,j]` has no c term when c is not a
pivot. Measured 0.0 on 8/8 draws, and the operator entry `A[i,j]` does not move
either.

SUCCESSOR: **`pivot_signed__random_P`** -- identical k, identical operator,
identical intervened positions, P drawn uniformly at random instead of by
key-norm. `scale/s2_probe.py::random_pivots` already exists and is already bound
by `test_s2_ablation.py::test_random_pivots_are_content_blind_and_not_the_key_norm_set`.
Why it survives: it is nonzero-capable by construction -- c is drawn from P, so
the c term is present -- and it isolates the one variable the frozen claim
actually names, CONTENT-SELECTION, from the one that trivially produces flatness,
|P| = k fixed. `dense_signed__at_pivots` stays as the routing control.
**Pre-registered kill:** slope(random P) > -0.3 AND the Clopper-Pearson 95%
intervals of random-P and content-P overlap at every s >= 512. Then selection
contributes nothing, the result is fixed path count, and that is
Star-Transformer 1902.09113 (2019). Claim sentence rewritten before work resumes.

This is also G4's successor. G4 as written compares against `pivot_unsigned`,
which is softmax-based and therefore structurally 0 by the same semiring theorem
M1 states -- it can never fire. The comparison that CAN fire is against
random-P, not against unsigned.

## M4 -- `max change > 1e-12`

DEAD, and doubly so. `lowest_salience_token(..., exclude=keep)` guarantees the
perturbed token is outside `keep`; `settle_evicted` reads `x[keep]` and nothing
else. Measured: token inside `keep` in 0/8 draws, change exactly 0.0 in 8/8.
The second tautology: `settle_evicted` computes
`settle_exact(evicted_operator(x, keep), x[keep])` where `evicted_operator` is
`softmax(scores(x[keep]))` -- so it IS `settle(x[keep])` by definition. The test
compares a thing to itself by two names.

SUCCESSOR: **path-equivalence against an INCREMENTAL evictor.**
`max | settle_incremental(state, evicted_token) - settle_from_scratch(x[keep]) |`,
where `settle_incremental` updates an already-settled state in place. The two
sides are then built by different routes and the equality is a claim rather than
a restatement.
Why it survives: the quantity can move. Any retained row sum, any cached
denominator, any stale normalizer shows up as a nonzero difference -- which is
precisely the failure M4 exists to exclude, and precisely what the gated arm
does (2.3e-3 deficit, measured, real).
**Pre-registered kill:** difference > 1e-12, OR **no incremental evictor exists
in the shipped module** -- in which case M4 is RED BY ABSENCE, not GREEN by
identity. Liveness clause: the same comparison run against `gated_operator` must
exceed 1e-12, or the detector is uncalibrated and M4 is UNEVALUABLE.

## M2's `windowed` arm

DEAD: `build_arm` never passes `window=`. The label selects only the pool `c` is
drawn from while hop 2 stays routed through the global P, so the cell measures
P(window token is a pivot) -- 88/320, 18/320, 3/320 at s=32/128/512.

SUCCESSOR: **window the OPERATOR.** `ceq/bench.py` already takes `window=` on
every builder; the probe simply never uses it. Sweep w in {8, 64, 256} and
report the PAIR (slope, reach), reach = max |i-j| carrying nonzero influence.
Why it survives: it measures the two quantities the frozen baseline sentence
actually asserts a trade between.
**Pre-registered kill:** a windowed operator at w << s that is flat AND reaches
comparably to the pivot arm falsifies "windowed arms buy flatness only by
surrendering reach", and that sentence leaves the claim.

## M2's operator -- the scope, not the number

DEAD AS SHIPPED: the arm builds A with `_causal_tgate_operator`, which grep
finds in `ceq/bench.py` and nowhere else. The Hub ships `sgate`;
`ceq/attention.py` ships `rho*w/||w||_1`. Both shipped forms carry the
denominator `ceq/bench.py:242` names as the measured cause of the 1/s death.

SUCCESSOR: **`pivot_sgate`** -- pivot routing on the operator that actually
ships. Two lines in `build_arm`'s dispatch; the operator already exists.
Why it survives: it is the only version of M2 whose result can appear in a model
card describing this package.
**Pre-registered kill:** slope(pivot-routed sgate) < -0.3. Then flatness is a
property of denominator-freeness rather than of routing, and no M2 sentence may
be written about the shipped module until the module IS the denominator-free one.

## M3, M6, S3

M3: kill clauses are sound; the instrument is not. `bootstrap_ci` has zero
callers and no softmax arm exists, though M3's own text requires the softmax arm
to run FIRST and its failure distance to be recorded before ours. Successor is
the same kill with the ordering enforced mechanically rather than by intention.

S3: no file implements the Carnot grading, so "derived beats learned" has no
left-hand side. Successor: implement it or strike S3 and report the geometry as
structure, which S3's own text already names as the fallback.

M6: **THIS NEEDS A LEAP, not more evidence.** No module computes a numerical
radius, so the guard the kill reads does not exist -- but building it does not
resolve the item. `ARSENAL.md:71` records w(A) = 1.499315 and calls the bound
vacuous, while the denominator-free operator's stated price is row L1 growing as
O(s*g), so hop-2 magnitudes scale with s. Magnitude control and context-flatness
are pulling opposite ways, and I cannot name a successor I can defend: any
scalar bound tight enough to be worth publishing may be exactly the denominator
whose absence M2's flatness depends on. What a successor must do: bound
||A^h|| by a quantity with NO sum over s, without reintroducing a per-row
normalizer -- or show that the two goals are formally incompatible, which is
itself a publishable result and would fold M6 into M2 rather than leaving it
standing alone.

### S2, RESTATED — and K5 pre-registered before the arm exists (Cameron)

**S2 as written is unanswerable, and the reason is a theorem, not a shortage of
draws.** It names three legs — "pivots+unsigned vs pivots+signed vs
dense+signed". The unsigned leg is at an exact structural floor on the only
metric that expresses the capability (K2, measured above: 0/1024 at every size,
CP95 [0, 0.00360]). Two arms at exactly zero separate nothing.

**But the degeneracy is narrower than "softmax cannot", and this repository
already measured how much narrower.**
`tests/cameron/test_negation_is_the_axis.py::test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`
shows **one GELU between two softmax layers restores the sign flip**. The
semiring theorem constrains non-negative operators with LINEAR value paths. A
real transformer has an MLP. So the non-degenerate unsigned arm is not a
contrivance that has to be invented — it is **softmax with a nonlinearity between
the hops**, which is not an approximation of Star-Transformer 1902.09113 but
literally its update: satellites gather to the relay, the relay is transformed,
the relay scatters back.

    r    = A[P,:] v          gather to the k pivots
    r    = gelu(r)           the nonlinearity a real block has
    out  = v + A v + A[:,P] r    scatter back

With `gelu` replaced by the identity this is exactly `v + Av + (A[:,P]A[P,:])v`,
the probe's existing pivot forward — which is what makes it bindable rather than
a second implementation that merely agrees.

**S2 restated.** It is not a 3-list. It is three binary axes read on two metrics:

| axis | levels |
|---|---|
| ROUTING | hop 2 through k=8 fixed pivots / dense over all s |
| SELECTION | pivots content-selected (key-norm) / uniformly random |
| SIGN | signed operator / non-negative operator |

| metric | what it expresses | who can score on it |
|---|---|---|
| `rate` | context-conditional sign capability | signed operators, and non-negative ones ONLY through a nonlinearity |
| `R = term/sigma` | dilution-resistance (the small-ball share) | every arm, signed or not |

#### K5 — pre-registered now, before `pivot_softmax_gelu` exists

Arm: softmax base matrix, hop 2 routed through k=8 content-selected pivots, GELU
at the hub between gather and scatter, c drawn from P, PROTOCOL SCALING, sizes
8/32/128/512. Control: `dense_softmax_gelu`, same nonlinearity, dense hop 2.

| id | condition | consequence |
|---|---|---|
| **K5a** | `pivot_softmax_gelu` rate > 0 at every s **AND** slope > -0.3 | **G4 FIRES ON ITS INTENDED MEANING.** A non-negative operator carrying the nonlinearity a real transformer already has holds the sign-flip rate flat under pivot routing. The capability lives in the unsigned ablation once the ablation is made non-degenerate. The result belongs to Star-Transformer 2019 and the claim must be rewritten to depth-and-parameter efficiency at ONE layer — which is the restatement this repo was already forced into once. |
| **K5b** | `pivot_softmax_gelu` rate = 0 at every s | the GELU escape does not survive pivot routing; the arm is degenerate again; **K5 is VOID** and no substitute metric may be introduced to rescue it |
| **K5c** | `pivot_softmax_gelu` flat **AND** `dense_softmax_gelu` also flat | routing is not what holds the rate up in the unsigned case; the routing mechanism does not transfer across the sign axis, and that is a finding about the mechanism rather than about the arm |
| **K5-bind** | the same forward with `gelu` replaced by the identity must reproduce `pivot_probe.run_arm`'s `rate`, `k`, `n`, `term`, `sigma` **bit-identically** for both the signed and the softmax arms | otherwise the arm is a second implementation and every number from it is void (G3) |

**The prediction, stated before the run so that it can be wrong:**
`pivot_softmax_gelu` will read NONZERO and FLAT, because gather-scatter through k
pivots gives the same 1/k share the signed arm gets, and the GELU supplies the
sign. If that is what comes back, **K5a fires and the capability claim collapses
to depth-and-parameter efficiency** — the same place the GELU finding put it
before, now with the routing axis included.

**This is the refusal of the stated tradeoff, and it is not optimism about the
outcome.** The framing offered was "signedness wins" or "S2 is unanswerable".
Both are false: the third measurement exists, it is not degenerate, it is a
published architecture's own forward pass, and it decides the question in either
direction. An optimist's job is to find the measurement that can be made, not to
find the answer that is liked.


---

## LOOP ITERATION 27 — 2026-08-25 — s=8 COLLISION FIXED. Rerun at |P|=8 via s=16.

CALIBRATION [RUN] exit 0, 4/4 bit-identical.

**THE FIX, and why it took the form it did.** The user asked to rerun s=8 at
|P|=8. **That is impossible and the reason is structural:** `select_pivots`
excludes i and j, so at s=8 with i=7, j=2 only SIX indices remain -- |P| = 6, not
the k=8 every artifact claimed. Measured [RUN] across the sweep:

    s      i    j   cands  |P|  intermed  in P  excluded   |pivot-dense| at [i,j]
    8      7    2       6    6         4     4         0   1.863e-09  <- DEGENERATE
    16    15    4      14    8        10     6         4   6.646e-03
    32    31    8      30    8        22     6        16   9.546e-02
   128   127   32     126    8        94     6        88   2.180e-01

Routing bites only when some intermediate strictly between j and i is NOT a
pivot, i.e. when **k < i-j-1 = s - s//4 - 2**. For k=8 that needs **s > 13.3**.
So s=8 cannot host |P|=8 at all, and **s=16 is the smallest size where |P|=8 is a
genuine proper subset** -- 4 intermediates excluded, delta 6.6e-03, four orders
above the probe's 1e-6 floor.

**s=8 is marked DEGENERATE** for the collided pair and excluded from `units()`
and from every fit. Its journalled units stay on record as the degenerate point
they are, and `--status` now NAMES them rather than counting them:
`(3 journalled but NOT in the plan: [...])`. **s=16 added.**

**[RUN] THE RERUN, |P|=8 genuine:**

    s=16  pivot 0.028320 (116/4096)   dense 0.025879 (106/4096)   ratio 1.09x

**They are CLOSE but NOT identical** -- unlike s=8, where they agreed bitwise to
every digit (both k=101, n=4096). That is the collision gone: at s=16 the two
arms are measuring different things, and the ratio starts near 1 exactly as it
should when routing has only just begun to restrict.

**CORRECTED SLOPES over s=16..2048** -- pivot **+0.0084**, dense **-0.8719**.
The dense slope was **-0.746** while the degenerate s=8 point anchored the fit.
**The separation is real and gets STRONGER once the collision is removed**, which
is why this correction costs the claim nothing and costs the wrong number
everything.

**THE REPLAY ROTATION WORKED [RUN]:** this bucket replayed
`pivot_signed__in_P/s32`, not the `s8` cell it had replayed for nine consecutive
buckets. The determinism audit now walks the set instead of re-checking one
four-second cell forever.

**STANDING ORDER ISSUED to Foreman, Cameron and Chase**, per the user: widen from
their single items to ALL SIX M points, and **a disproof must arrive with its
successor** -- the patient cannot be let to die. A RED still stops BUILD work on
that item; what is added is that naming the replacement is part of the
deliverable. **Dr House is on standby** for the case where an item dies and the
way forward needs a leap rather than more evidence.

**THREE MORE NURSE ROLES under Chase**, per the user: INFERENCE engineer (does
pivot routing help at DECODE, where one query row is live and
`A[i,P] @ A[P,:]` is O(ks) against dense O(s^2)?), SYSTEMS engineer (roofline;
is the 535 ms/draw control FLOP-bound or bandwidth-bound?), and
REPRODUCTION-PROTECTION engineer (what tooling would have caught SIXTEEN
internally-consistent-externally-wrong instruments?). Plus a directed scan --
using the `github-master` API key -- of **vLLM, Kimi K2/K3, MoBA, DeepSeek NSA
and flash-linear-attention** for engineering tactics this project lacks, with one
specific question: **how does a shipped inference stack verify a new attention
kernel is correct?** This repo has 847 tests it cannot run to completion; those
repos gate merges on CI.

CHECKLIST: no status changed. M2 **38/137**, UNTESTED.

---

# CHASE — ENGINEERING PASS. Ranked by measured payoff.

## 1. The M2 measurement never needed the matrix it spends all its time building

`run_arm` reads exactly one number per branch: `float(grad[j].sum())`. Since
`a` and `hop2` never depend on `v`, `h = M v wo` is linear and

    grad[j].sum() == (delta_ij + A[i,j] + hop2[i,j]) * wo.sum()

exactly. So dense hop2 is `A[i,:] . A[:,j]`, **O(s)**, and pivot hop2 is
`A[i,P] . A[P,j]`, **O(k)**. The `[s,s]` product is built and thrown away.

Where the time actually goes at s=2048 -- the O(s^3) assumption was right for
the wrong reason, and the operator is not the problem:

| stage | ms |
|---|---|
| `_causal_mask` | 3.09 |
| `qh @ kh.T` | 3.56 |
| `tanh` | 5.04 |
| `masked_fill(~m)` | 12.96 |
| **full tgate operator** | **29.61** |
| **`a @ a`** | **444.32** |

| arm | s | now ms/draw | closed form | speedup |
|---|---|---|---|---|
| dense | 512 | 26.2 | 15.4 | 1.7x |
| dense | 1024 | 56.3 | 26.6 | 2.1x |
| **dense** | **2048** | **735.7** | **68.1** | **10.8x** |
| pivot | 2048 | 118.4 | 68.1 | 1.7x |

**It is drop-in, and this is the number that decides it.** `term` and `sigma`
are O(s) reads that never touched the matmul, so they are bitwise identical;
`rate`/`k`/`n` matched exactly at every configuration tried (mine 3, a second
pass 6 more, ~2000 draws, zero disagreements). The residual risk is a draw
sitting within the paths' 3e-5 relative arithmetic difference of the 1e-6 flip
floor. Measured: the closest of 512 draws is **361x the floor**, and **zero**
draws come within 100x. `bucket.py`'s replay audit passes.

DERIVED, from measured ms/draw against the current unit list: remaining M2 drops
from ~4.4 h to ~1.9 h, **53 buckets to 23**.

## 2. Training had no resume path; now it does, bit-exact

`ceq/hf/train.py::train()` saved weights only -- no optimizer moments, no step,
no RNG. Without it the largest trainable model is whatever fits one
uninterrupted free-tier session. Patched with `resume_from` + `trainer_state.pt`
(optimizer state, step, torch RNG, the data generator's state -- which IS the
dataloader position, since batches are drawn by index and no shuffling loader
exists). No scheduler exists to save: `lr` is a constant.

    uninterrupted 2N : [5.511620, 5.515772, 5.578649, 5.495078, 5.587345, 5.551126]
    split N + resume : [5.511620, 5.515772, 5.578649, 5.495078, 5.587345, 5.551126]
    max abs diff     : 0.0

`tests/chase/test_resume_checkpoint.py` 1 passed, 67.57 s. Hardened after the
fact: the checkpoint load was `weights_only=False` on untrusted pickle input.

## 3. A suspicion worth refuting: the softmax baseline is NOT handicapped

C6 demands the comparison run against a tuned baseline. It does.
`ceq/lm.py:175` -- `F.scaled_dot_product_attention(q, k, v, is_causal=True)`.
`ceq/sizing.py:164` -- the softmax arm's analytic activation term is linear in
S with no `S^2` term, the signature of a fused kernel that never materializes
`[S,S]`. The COSTS memory and wall-clock ratios stand.

## 4. Tried, did not pay off -- with the numbers

Caching `_causal_mask` (it rebuilds `ones(s,s).tril(-1)` every call) and
replacing `masked_fill(~m)` with a bool multiply is **bitwise identical** and
worth 23% at s=1024 (9.93 -> 7.60 ms) but is a **wash at s=2048** (25.01 ->
25.12 ms). Memory-bandwidth bound once out of cache. Not worth doing at the size
that matters.

## 5. The streaming question, answered by measurement

The worry was that `tanh` has no rescaling identity, so there is no
FlashAttention analogue and this operator's memory cost is structurally worse
than softmax's. **That is backwards.**

Online softmax's running max and rescaling exist for exactly one reason: to
compute a row-global DENOMINATOR without seeing the whole row. `A_ij = g_i *
tanh(qhat_i . khat_j / tau)` has no denominator, so each entry is computable
from `(q_i, k_j)` alone and tiling needs no rescaling at all. Hop 2 costs one
extra sequential pass, not a materialized matrix: build the k pivot rows, form
`u = A[P,:] v` (a k x d object), then stream row-tiles accumulating
`A_tile v + A_tile[:,P] u`.

Measured, against the dense path:

| s | max rel err | dense [s,s] | streaming peak | reduction |
|---|---|---|---|---|
| 512 | 8.83e-08 | 1.00 MiB | 0.25 MiB | 4x |
| 1024 | 3.78e-07 | 4.00 MiB | 0.50 MiB | 8x |
| 2048 | 3.46e-07 | 16.00 MiB | **1.00 MiB** | **16x** |

Error is fp32 roundoff. The reduction is `s/tile` and grows with context. The
`[B,H,S,S]` materialization in `COSTS` is an implementation choice, not a
property of the operator, and the model card should not carry it as one.


---

## LOOP ITERATION 30 — 2026-08-25 — S2's DECISIVE ARM IS DECAYING. G4 does not fire.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.

**Reordered S2 so the decisive arm runs FIRST.** `pivot_unsigned__x` is what G4
turns on, and it was scheduled last, behind a 16384-draw `pivot_signed__x/s1024`
unit that overran the window. Order only; nothing cut.

**[RUN] THE COMPARISON, matched geometry, matched pivots, matched draws:**

    s                    8          32         128
    pivot_unsigned__x  0.102539   0.014160   0.001221    slope **-1.598** (R2 0.9962)
    pivot_signed__x    0.026367   0.034912   0.024902    slope **-0.021** (R2 0.0250)
    k/n                420/4096   58/4096    5/4096      (unsigned)

**Unsigned routing falls 84x over a 16x context growth. Signed routing is
flat.** This is pre-registered **outcome 2**: signedness is load-bearing ON TOP of
routing, and that combination is the unoccupied cell. **G4 DOES NOT FIRE.**

**AND NOTE WHICH ARM STARTS HIGHER.** At s=8 the UNSIGNED arm reads 0.1025
against the signed arm's 0.0264 -- softmax flips the sign of j's influence on i
nearly FOUR TIMES MORE OFTEN at short range, then collapses. The signed operator
is not better at s=8; it is better at HOLDING the property. That is a sharper and
more defensible statement than "signed attention has a property softmax lacks",
and it is the opposite of what the project claimed for most of its life.

**WHAT IS NOT ESTABLISHED YET.** The unsigned arm has run at s = 8/32/128 only;
s = 512/1024/2048 are pending, and the signed arm already has s=512 at 0.031250.
Three sizes is enough to see a slope of -1.598 against -0.021, and it is not enough to
close G4 -- `report()` still requires every cell, and it prints the G4 verdict
itself rather than leaving it to a reader.

**Also unchanged by this:** it is still a MECHANISM result, not a capability
(M3 UNTESTED), still on random projections, and still measured on a property one
GELU between two softmax layers restores.

CHECKLIST: no status changed. M2 35/132; S2 11/42.

### K5 — RUN. The prediction in the pre-registration was WRONG, and that is the point.

`tests/cameron/test_s2_ablation.py` 9/9 GREEN, each preceded by a recorded RED.
The K5-bind passed: `run_hub_arm` with `nonlin=None` reproduces
`pivot_probe.run_arm` **bit-identically in all five fields on four arms**
(`pivot_signed`, `dense_signed`, `pivot_unsigned`, `dense_unsigned`), so the
separate forward is admissible.

**Arm as pre-registered** — `bench._softmax_operator`, gather → GELU → scatter
through k=8 content-selected pivots, n=1024, c in P, SCALING:

| s | rate | CP95 | flips |
|---|---|---|---|
| 8 | 0.029297 | [0.01985, 0.04156] | 30 |
| 32 | 0.002930 | [0.00060, 0.00854] | 3 |
| 128 | 0.000977 | [0.00002, 0.00543] | 1 |
| 512 | 0.000000 | [0.00000, 0.00360] | 0 |

**slope(rate) = -1.227.**

**De-confounded, post-hoc and labelled as such.** The pre-registered arm uses the
near-argmax softmax (participation ratio 1.15-1.51). `matched_softmax_operator`
scores `qhat.khat/tau` — the SAME scores `tgate` uses — and is a genuine mixture:
participation ratio **6.62 / 29.12 / 119.92 / 484.51** at s = 8/32/128/512, i.e.
tracking s. Same gather-GELU-scatter, n=1024:

| s | rate | CP95 | flips |
|---|---|---|---|
| 8 | 0.008789 | [0.00403, 0.01662] | 9 |
| 32 | 0.000977 | [0.00002, 0.00543] | 1 |
| 128 | 0.000000 | [0.00000, 0.00360] | 0 |
| 512 | 0.000000 | [0.00000, 0.00360] | 0 |

**slope(rate) = -1.585.** Dense matched control, n=512: 0.013672 at s=8, then 0
at 32/128/512 (1/4 nonzero).

| kill | fires? | number |
|---|---|---|
| **K5a** | **NO**, on both comparators | slope -1.227 (pre-registered) and -1.585 (de-confounded), both far below the -0.3 bar |
| **K5b** | **NO** | the arm is non-degenerate — 30 flips at s=8 unmatched, 9 matched. It can score, and it loses |
| **K5c** | **cannot fire** | its condition needs the pivot arm flat; it is not |
| **K5-bind** | **HELD** | five fields bit-identical to `run_arm` on four arms at identity |

**THE PREDICTION WAS WRONG AND IT WAS WRITTEN DOWN FIRST.** The pre-registration
said, in terms: "`pivot_softmax_gelu` will read NONZERO and FLAT ... K5a fires and
the capability claim collapses to depth-and-parameter efficiency." It read nonzero
and **steeply decaying**. A pre-registration that only ever confirms is decoration;
this one cost its author the outcome he argued for, which is the only evidence that
the mechanism works.

**What that buys, stated exactly.** The GELU finding
(`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`) showed a
nonlinearity RESTORES the sign flip, and was correctly read as forcing the claim
down to depth-and-parameter efficiency. It restores **existence**. It does not
restore **context-stability**: with the nonlinearity in place and the hop routed
through k fixed pivots — Star-Transformer 1902.09113's own update, verbatim — the
rate still dies at -1.585, from 0.0088 to exactly 0 across a 64x context growth,
while the signed operator holds +0.027 across 256x. **Sign existence and sign
stability are different properties, and only the first one a nonlinearity buys.**

**G4 DOES NOT FIRE, now on a metric the unsigned arm can score on.** That was the
hole K2 left: two arms at exactly zero separate nothing. This arm is not at zero.

---

### THE SCOPE RED — the operator every M2 number describes SHIPS NOWHERE

`tgate` lives in `ceq/bench.py` alone. The Hub config ships `sgate`;
`ceq/attention.py` ships the row-L1 form. Both carry a denominator that sums over
context — which `_causal_tgate_operator`'s own docstring names as "the measured
cause of the 1/s death". So the same routed measurement was pointed at the
SHIPPED matrix, bound by
`test_the_shipped_operator_swap_really_swaps_the_shipped_operator` (asserts the
swap equals `bench._causal_sgate_operator` exactly, differs from tgate, and that
hop 2 is `A[:,P] A[P,:]` on the swapped operator).

`sgate`, routed hop 2, k=8, c in P, n=1024 (dense control n=512), SCALING:

| s | pivot rate | CP95 | dense rate | CP95 |
|---|---|---|---|---|
| 8 | 0.154297 | [0.13270, 0.17790] | 0.152344 | [0.12232, 0.18644] |
| 32 | 0.023438 | [0.01507, 0.03467] | 0.035156 | [0.02097, 0.05499] |
| 128 | 0.000977 | [0.00002, 0.00543] | 0.001953 | [0.00005, 0.01083] |
| 512 | 0.000000 | [0.00000, 0.00360] | 0.001953 | [0.00005, 0.01083] |
| **slope** | **-1.826** | | **-1.151** | |

**M2's pre-registered kill is `slope(c in P) < -0.3`. On the operator that
actually ships, the routed arm reads -1.826 — the kill fires by a factor of six.**
And routing makes it **worse**, not better: -1.826 routed against -1.151 dense.

**The mechanism, and it is the one `bench.py` already named.** Pivot routing fixes
the PATH COUNT. It does not remove a DENOMINATOR. `sgate` normalizes over the
whole prefix, so a token's leverage is its share of a sum that grows with s no
matter how few paths reach it — and restricting hop 2 to k pivots removes paths
from the numerator while leaving the denominator summing over all s, which is why
the routed arm decays faster than the dense one. `tgate` has no such sum. So the
flat +0.027 is a property of DENOMINATOR-FREEDOM plus fixed path count, and the
module ships neither.

**Consequence, which is a scope statement and not a retraction.** Every M2 number
in `results/m2.jsonl` remains correct about `tgate`. None of them is about the
module on the Hub. Either `tgate` ships, or the claim sentence must name the
operator it is about — and `README.md` / `MODEL_CARD.md` have been updated to say
so rather than leaving the reader to assume.

---

### THE SIX M POINTS — every kill paired with its successor

Per the standing rule: whoever kills an item owns naming what stands in its place.
Successors are **PROPOSED** — pre-registered kills, not measurements. Nothing below
is claimed as run except where a number is given.

| id | status after this run | why | **successor, with its pre-registered kill** |
|---|---|---|---|
| **M1** | precondition, and NARROWER than stated | softmax+GELU reaches negative influence; the semiring theorem covers non-negative operators with LINEAR value paths only | **M1' — context-STABLE negative influence.** Existence is free once a nonlinearity is present (measured: 30 flips at s=8). Kill: signed-operator rate slope < -0.3, OR any non-negative-plus-nonlinearity construction whose CI overlaps the signed arm at s >= 128. Measured this run: the matched Star-Transformer arm is at exactly 0 there, CP95 [0, 0.00360] |
| **M2** | survives on `tgate` (+0.027); **RED on the shipped operator (-1.826)**; "content-selected" struck (K4, +0.081) | routing fixes path count, not denominators; and M2 draws c FROM P so it conditions content-selection away | **M2' — measured on the SHIPPED operator, unconditionally.** Kill: slope < -0.3 on whatever operator the Hub config actually instantiates. Already fires at -1.826 for `sgate`, so the successor's real content is: **ship `tgate`, then re-run.** For the selection half: `rate x P(c in P)` with c designated by the TASK; kill: if that recall decays like k/s for the key-norm selector, content-selection is dead permanently |
| **M3** | never run; not killed, absent | the bridge from operator property to capability | unchanged, and now the ONLY place content-selection can be tested — it is the only instrument with a token designated from outside the selector. This needs no successor, it needs running |
| **M4** | kill is a **tautology** | `lowest_salience_token(exclude=keep)` puts the perturbed token outside `keep` by construction; 0/8 inside, 8/8 exactly zero | **M4' — the evicted token is designated from OUTSIDE the salience function**, drawn uniformly from the non-keep set, and the read must include tokens the salience function WOULD have kept. Kill: max change > 1e-12 on a uniformly-drawn evictee; and the unit is VOID, not GREEN, if the draw is still made by the mechanism under test |
| **M5** | half-bound | theorem at `N = n`, module ships `hops = 2`, grep-binding contains "hops" zero times | **M5' — bind the theorem to the shipped TRUNCATION.** Either ship `hops = n`, or restate as "exact at hops=2 over the class where the tail is negligible" and MEASURE the tail: `\|\|A^3 v\|\| / \|\|out\|\|`. Kill: tail above a pre-registered epsilon means the finite-resolvent certificate does not describe the shipped tensor |
| **M6** | guard does not exist; bound vacuous | `w(A) = 1.499315 > 1`, so `\|\|A^h\|\| <= 2 w(A)^h` grows | **M6' — enforce `w(A) <= 1` in the forward and re-run M2 under it**, which is M6's own stated test. Kill: the guard reintroduces decay in M2 (slope < -0.3), OR training diverges at every lr in the sweep. If it reintroduces decay, the denominator-free story is false and the successor is a per-hop MEASURED growth curve with no certificate claimed |

**THE PATTERN ACROSS THREE OF THE SIX, and it is one bug, not three.** M2's
`c not in P` clause, M2's content-selection claim, and M4's kill all draw the
quantity under test FROM the mechanism under test. `c` is drawn from `P` and then
`P` is credited; the evicted token is chosen by the salience function and then the
salience function is credited; `c not in P` is structurally invariant so the
control can never be nonzero. **The general repair is one sentence: the designated
token must come from the task, never from the selector.** That is why M3 is not
merely the next item — it is the only instrument in the programme that has one.

**One item needs a leap rather than more evidence, and it is not mine to
manufacture.** If `tgate` cannot be made to ship — if the denominator is load-
bearing for training stability in a way `bench.py` never had to face, since it
never trained anything — then the whole M-programme is about an operator that
cannot exist in the module, and no additional measurement resolves that. **DR
HOUSE:** the question is whether a denominator-free operator can be trained at all
at 25M+ parameters, or whether row-L1 growth (the 1.5e4x spread already on record)
makes it diverge. That is an existence question about training dynamics, not a
probe question, and it decides whether M2 has a subject.

#### ROBUSTNESS: every slope above, recomputed WITHOUT the degenerate s=8

s=8 with k=8 is one arm reported twice — after excluding i and j only 6 candidates
remain, so |P| = 6 not 8 and the pivot and dense forwards coincide. Every slope in
this section was computed including it. Recomputed without it (zeros dropped, never
clamped):

| arm / metric | incl s=8 | pts | EXCL s=8 | pts |
|---|---|---|---|---|
| `pivot_signed` rate | +0.027 | 6 | **+0.004** | 5 |
| `dense_signed` rate | -0.746 | 5 | **-1.009** | 4 |
| `randpivot_signed` rate | +0.081 | 4 | **+0.083** | 3 |
| `pivot_softmax_gelu` rate | -1.227 | 3 | **-0.792** | 2 |
| same, MATCHED softmax | -1.585 | 2 | **n/a** | 1 |
| `pivot_SHIPPED_sgate` rate | -1.826 | 3 | **-2.292** | 2 |
| `dense_SHIPPED_sgate` rate | -1.151 | 4 | **-1.043** | 3 |
| `pivot_signed` R | -0.038 | 6 | **-0.016** | 5 |
| `dense_signed` R | -0.913 | 6 | **-0.945** | 5 |
| `pivot_absmag` R | -0.046 | 4 | **-0.063** | 3 |
| `dense_absmag` R | -0.930 | 4 | **-1.013** | 3 |

**`dense_signed` lands at exactly -1.009, independently reproducing the correction
made elsewhere this round from a different code path — which is the only reason to
trust the rest of the column.** Every verdict survives: K4 still fires (+0.083),
K5a still does not (-0.792, bar is -0.3), the routing-vs-sign decomposition on R is
unchanged (routed -0.016/-0.063 signed/unsigned against dense -0.945/-1.013), and
the scope RED gets worse, not better (-2.292).

**THE ONE THING THAT DOES NOT SURVIVE, STATED:** the MATCHED
`pivot_softmax_gelu` arm has only one nonzero cell left once s=8 is dropped, so its
**slope is unevaluable without s=8**. What survives there is not a slope but a
floor: 0 flips in 1024 draws at s=128 and s=512, **CP95 [0.00000, 0.00360]**,
against the signed arm's 0.0288-0.0310 at those sizes. That is a bound, not a fit,
and it is the form the K5a verdict should be quoted in.

---

## FOREMAN — what is actually producing the flat 0.03 (and what it means for M1-M6)

### Root cause, one sentence

The M2 flip statistic is `sign(A_ij + H_ij)`, so the plateau is the small-ball
density of the **one-hop direct edge** `A_ij` — a quantity with no `s` and no `k`
in it — and pivot routing contributes nothing to it: deleting the entire k-term
routed sum leaves the rate at 0.0264 against 0.0310 full (RED, n=4096).

### Generalised, in the project's own idiom

Prior root causes read "an exclusion rule not invariant under the same group as
the statistic it gates". The M-programme version is one rung more general:

> **The harness varies a quantity that lies in the KERNEL of the map it then
> measures, so what gets journalled is a constant of construction wearing the
> shape of a measurement.**

Instances, all with both halves quoted:
1. M2 clause 2 — `c` drawn from outside `P`; statistic sums only over `P`. delta ≡ 0.
2. M4 — crushed token excluded from `keep`; `settle_evicted` reads `x[keep]` only. delta ≡ 0.
3. M2 selector — `select_pivots` ranks on `key.norm`; `_causal_tgate_operator`
   L2-normalises keys. **RED: 7.5x key rescale moves the operator by 2.98e-08.**
4. `placement="windowed"` — `build_arm` never passes `window=` to any operator;
   the label changes only the pool `c` is drawn from (`scale/pivot_probe.py:171`
   vs `:122`). Held by `tests/chase/test_m2_instrument_binds.py:121`.
5. The published windowed-attention claim sweeps `s` at fixed `w=8`, but at fixed
   `w` the statistic reads only `[i-16, i]` and is constant in `s` by construction.
6. My own `scale/mech_attack.py` and `scale/s2_probe.py` reproduce instance 1's
   shape (`placement="not_in_P"` + `a[i,P]*a[P,j]`). No caller uses it. Flagged.

### The nine REDs

| test | pre-registered claim | measured |
|---|---|---|
| `test_pivot_plateau_scales_with_k::..._k_to_the_minus_half` | rate(32)/rate(8) ∈ [0.35,0.71] | **1.222** (27/1024 vs 33/1024) |
| `test_m2_mechanism_story::..._term_dilutes_as_one_over_s` | slope(term vs s) ≤ -0.5 | **+0.0225**, R²=0.519 |
| `test_m2_mechanism_story::..._decay_is_a_power_law` | first3/last3 slope ratio ∈ [0.5,2] | **0.230** |
| `test_m2_mechanism_story::..._background_is_what_the_plateau_measures` | \|FULL−NO_BG\|/FULL > 0.25 | **0.150** |
| `test_m2_mechanism_story::..._one_hop_is_not_what_the_plateau_measures` | \|FULL−NO_ONEHOP\|/FULL ≤ 0.25 | **2.276** |
| `test_pivot_selector_is_content_bearing::..._operator_sees_the_key_norm` | max\|ΔA\| > 1e-6 under 7.5x key rescale | **2.98e-08** |
| `test_pivot_selector_is_content_bearing::..._beats_random_selection` | rel diff > 0.25 | **0.0315** (127/4096 vs 131/4096) |

### The ablation table (RUN, `scale/mech_attack.py` + scratchpad `ablate.py`)

s=128, pivot_signed, seed 0. Rates from the SAME draws, four counterfactuals:

| k | FULL | NO_ONEHOP | NO_BACKGROUND | NEITHER | std(a_ij) | std(bg) |
|---|---|---|---|---|---|---|
| 2 | 0.0278 | 0.2633 | **0.0283** | 0.3650 | 0.1290 | 0.0138 |
| 8 | 0.0287 | 0.1028 | **0.0267** | 0.3643 | 0.1290 | 0.0457 |
| 32 | 0.0261 | 0.0352 | **0.0293** | 0.3640 | 0.1300 | 0.1270 |
| 64 | 0.0237 | 0.0256 | **0.0256** | 0.3684 | 0.1300 | 0.2223 |

s=512 k=8: FULL 0.0288 / NO_ONEHOP 0.1084 / NO_BG 0.0342.
s=1024 k=8: FULL 0.0400 / NO_ONEHOP 0.1113 / NO_BG 0.0420.
NO_BACKGROUND sits at 0.026–0.043 at **every s and every k in both arms**. That
is the invariant. `NEITHER` = 0.36 is the ceiling (P(t1,t2 opposite signs)).

Algebra behind it, verified to 5.7e-04 relative against autograd over 128
(draw,branch) pairs: `grad[j].sum() == (A[i,j] + hop2[i,j]) * wo.sum()`.
`A_ij` does not depend on `x[c]` at all, so the c-dependence is the single term
`A[i,c]A[c,j]`; everything else is a fixed offset the flip must cross.

### What -0.746 actually is

It is a contaminated fit. `s=8` is one arm reported twice (|P|=6, pivot≡dense
because the only dropped terms `p=i`,`p=j` vanish on a strictly lower-triangular
A). Dropping it moves the dense OLS slope **-0.746 → -1.009** (verified
independently). The corrected -1.009 is **minus the growth exponent of the
background's spread**: `sigma_bg ~ s^+0.9363, R²=0.9997`, the only real power law
in the dense arm, while `term ~ s^+0.0231, R²=0.541` — flat. So the dense arm does
NOT decay because a token's share dilutes as 1/s; the share never dilutes. It
decays because the background grows and swamps the one-hop offset.
Corrected caveat: with s=8 removed, the first3/last3 ratio is 0.498, i.e. the
"not a power law" RED is marginal, and the steepest segment (-2.700) rests on 2
flips in 4096. The *power law* claim survives at -1.0; the *mechanism* behind it
does not.

### 0.03 is not a structural constant. It is a `d`/`tau` constant.

| tau | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| rate | 0.0713 | 0.0498 | 0.0264 | 0.0117 | 0.0049 |

| d | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| rate | 0.0391 | 0.0439 | 0.0264 | 0.0186 | 0.0176 |

Local slope in tau at the operating point ≈ -1; in d from 8→64, ≈ -0.44.
Both match `rate ≈ f_{A_ij}(0) · E|Δt| ∝ tau^-1 · d^-1/2`. 0.03 is the value at
tau=1, d=16. It says nothing about routing.

### THE STRUCTURAL KILL: M2 does not survive the operator that ships

`_causal_tgate_operator` lives in `ceq/bench.py` and nowhere else. Re-running the
identical harness on `_causal_sgate_operator` (the Hub's shipped family):

| s | 16 | 32 | 128 | 512 | slope |
|---|---|---|---|---|---|
| pivot_sgate | 0.0366 | 0.0132 | 0.0005 | 0.0000 | **-2.12** |
| dense_sgate | 0.0264 | 0.0098 | 0.0010 | 0.0000 | -1.60 |
| pivot_signed (tgate) | 0.0278 | 0.0298 | 0.0322 | 0.0273 | -0.0023 |

**The pivot arm dies FASTER than dense on the shipped operator.** `mean|t1|` slope
is -6.47 for sgate against +0.018 for tgate: the denominator drives individual
entry magnitudes to zero, and routing cannot fix magnitude.

### THE RESCALE LEMMA, and the pre-registered prediction that confirmed it

A per-row scalar rescale λ maps the one-hop offset to `λ·offset` and the two-hop
probe term to `λ²·term`, so the small-ball probability — being the ratio —
scales as **λ**. Therefore:

> **A flat-in-s sign-flip rate is exactly equivalent to λ = Θ(1), i.e. to hop-2
> norm growing without bound. No operator has both.**

`ceq/bench.py:259-266` predicts that `static_scale` (divide row i by
sqrt(#visible keys) — position-only, no share reallocation) "is not expected to
reintroduce the decay". Prediction registered before running: *slope near -0.5,
mean|t1| like s^-1*. Measured, first run:

| s | 16 | 32 | 128 | 512 | slope |
|---|---|---|---|---|---|
| pivot_tgatex | 0.0088 | 0.0107 | 0.0029 | 0.0000 | **-0.5867** |

`mean|t1|` slope **-1.0269**; `std(a_ij)` falls as ≈ s^-0.63. bench.py's written
prediction is falsified, and the lemma is confirmed at three values of λ:
λ=1 → -0.002, λ=s^-1/2 → -0.587, λ=denominator → -1.74.

### Gate

`python run_calib.py --self-test` → exit 0, 4/4 bit-identical
(0.046875 / 0.0234375 / 0.1640625 / 0.0). Gate rejected a wrong target first.
No protected file was edited. New files only: `scale/mech_attack.py`,
`tests/foreman/test_pivot_plateau_scales_with_k.py`,
`tests/foreman/test_pivot_selector_is_content_bearing.py`,
`tests/foreman/test_m2_mechanism_story.py`.
Journal snapshot pinned: `results/m2.jsonl` 6355 bytes,
sha256 `c4d3c082a7d18085a9106f8e994e01e9ec89161e43d78cc538d11b0acc8628a6`
(it grew from 5658 bytes mid-session — another agent is appending concurrently).



---

## LOOP ITERATION 31 — 2026-08-25 — the unsigned arm is DEAD by s=512. Signed holds flat.

CALIBRATION [RUN] run_calib.py --self-test -> exit 0, 4/4 bit-identical.
Replay `pivot_unsigned__x/s8: MATCH`.

**[RUN] `pivot_unsigned__x/s512`: k=0, n=4096.**

    s                    8          32         128        512
    pivot_unsigned__x  0.102539   0.014160   0.001221   **0.000000**
    pivot_signed__x    0.026367   0.034912   0.024902    0.031250

Clopper-Pearson 95% upper bound on 0/4096 is **0.000731**, against the signed arm's
**0.031250** at the same s:

    separation at s=512  >=  **42.7x**   (a BOUND, not an estimate)

**BOTH ARMS ROUTED THROUGH THE SAME PIVOTS, SAME GEOMETRY, SAME DRAWS. The only
difference is signed versus non-negative entries.** Routing alone does not hold
the property: the unsigned arm has it at short range (0.1025 at s=8, FOUR TIMES
the signed arm) and loses it completely by s=512. The signed arm does not.

Supporting detail: the unsigned arm's `term` collapses to **2.06e-08** at s=512,
five orders below its s=8 value of 2.44e-02 -- the contribution through the
pivot has vanished, not merely been outvoted by noise.

**This is pre-registered outcome 2, now with four sizes.** G4 does not fire.
Signedness is load-bearing ON TOP of routing.

**WHAT IS STILL OWED.** s=1024 and s=2048 for the unsigned arm; `report()`
requires every cell and prints the G4 verdict itself. And every standing caveat
survives untouched: MECHANISM not capability (M3 UNTESTED), random projections
not trained ones, and a property that one GELU between two softmax layers
restores -- so the defensible claim remains depth/parameter efficiency and
PERSISTENCE IN CONTEXT, never "softmax cannot do this".

CHECKLIST: no status changed. M2 35/132; S2 12/42.


---

# ============================================================
# M2 IS **RED**. WORK-STOPPING CLAUSE IN FORCE.
# ============================================================

**2026-08-25. Killed independently by Foreman and Cameron, from different code
paths, and by the pre-registered k-sweep kill written before the arm existed.**

Per CHECKLIST.md: any MANDATORY item RED after its test runs => ALL BUILD WORK
STOPS. Only instrument repair, prior-art search, or write-up may continue.
**M2 bucket runs stop now.** 40/137 units; the remaining 97 would measure a
statistic now shown not to be about the mechanism.

## KILL 1 - the pre-registered k-sweep kill FIRED

Pre-registered: rate(k=32)/rate(k=8) in **[0.35, 0.71]** if the plateau tracks
k^-1/2. **Measured 1.222** (27/1024 vs 33/1024 at s=128). Conditional binomial
rejects k^-1/2 at **p ~ 4e-4**; 1/k is rejected far harder.

Sweep s=128, n=1024, k = 2..64: 0.0234, 0.0244, 0.0264, 0.0166, 0.0322, 0.0186 --
while sigma(background) climbs **0.0119 -> 0.2308**. **A 19x growth in the
background spread moves the rate not at all.**

## KILL 2 - the plateau is not the mechanism; ablation on the SAME draws

| k | FULL | NO_ONEHOP | **NO_BACKGROUND** | NEITHER |
|---|---|---|---|---|
| 2 | 0.0278 | 0.2633 | **0.0283** | 0.3650 |
| 8 | 0.0287 | 0.1028 | **0.0267** | 0.3643 |
| 32 | 0.0261 | 0.0352 | **0.0293** | 0.3640 |
| 64 | 0.0237 | 0.0256 | **0.0256** | 0.3684 |

**Deleting the entire k-term routed sum leaves the plateau intact.** Deleting the
ONE-HOP term multiplies it by 3.6x. NO_BACKGROUND sits at 0.026-0.043 at every s,
every k, in both arms.

**Root cause, one sentence:** the M2 flip statistic is sign(A_ij + H_ij), so the
flat 0.03 is the small-ball density of the **one-hop direct edge A_ij** -- a
quantity containing **neither s nor k** -- and pivot routing contributes nothing
to it. A_ij does not depend on x[c] at all.

## KILL 3 - the -1.009 dense slope is NOT a share dilution

sigma_bg ~ s^+0.9363 (**R2 0.9997**, the only genuine power law in the dense arm)
while term ~ s^+0.0231 (**R2 0.541**, flat). **The share NEVER DILUTES.** The
dense arm dies because the background GROWS. The entire "1/s share vs 1/k share"
framing -- this project's organizing story -- **has the wrong side of the ratio,
in both arms.**

## KILL 4 - content-selection contributes nothing (Cameron's K4)

randpivot_signed slope **+0.081**, CIs overlapping the content-selected arm at
every s. **"Content-selected" must be struck from the M2 sentence.** M2 draws c
FROM P, so it conditions content-selection away by construction. The identity was
already sitting unrun in scale/recall_probe.py:3-8.

Cameron's R = term/sigma separates on **ROUTING** (-0.04 vs -0.92) and is
**INDIFFERENT TO SIGN** (pairs agree to 0.008 / 0.017). **Fixed path count is
Star-Transformer's axis and it works without signs.**

## KILL 5 - M2 does not survive the operator that ships

| arm | slope | source |
|---|---|---|
| pivot_sgate | **-2.12** | Foreman |
| pivot_SHIPPED_sgate | **-1.826** (-2.292 excl. s=8) | Cameron |
| dense_sgate | -1.60 / -1.151 | both |
| pivot_tgate | -0.002 | both |

M2's bar is **-0.3**. **The kill fires by a factor of SIX on the operator that
actually ships**, and routing makes it WORSE than dense. Two agents, two code
paths, same verdict. tgate lives in ceq/bench.py alone; ceq/attention.py's
ceq_operator -- the one wired into register() -- is L1-normalized.

## THE RESCALE LEMMA - why this was never going to work

A per-row scalar rescale lambda sends the one-hop offset to lambda*offset and the
two-hop probe term to lambda^2*term, so the small-ball probability scales as
lambda. Therefore:

> **A flat-in-s sign-flip rate is EXACTLY EQUIVALENT to lambda = Theta(1), i.e.
> to an UNBOUNDED hop-2 norm.**

Pre-registered and confirmed first try: static_scale=True predicted "slope near
-0.5, mean|t1| like s^-1"; measured **-0.5867** and **-1.0269**. Three lambdas
agree: lambda=1 -> -0.002, lambda=s^-1/2 -> -0.587, lambda=denominator -> -1.74.

**ceq/bench.py:259-266 states in writing that static_scale "is not expected to
reintroduce the decay". It does.** So M2's claim reduces to *an operator whose
hop-2 magnitude grows like s has an s-independent sign-flip rate* -- close to a
tautology, and not shippable.

**M6 and M2 are the SAME EXPERIMENT.** Enforcing w(A) <= 1 and enforcing bounded
||A^2|| are both magnitude control, and the lemma says both kill the flat rate.

## THE GENERALIZATION - one sentence covering every kernel defect found

> **The harness varies a quantity in the KERNEL of the map it measures.**

Covers M2 clause 2 (c-not-in-P structurally zero), M4 (exclude=keep makes its
kill an identity), placement="windowed" (never passes window= to any operator --
scale/pivot_probe.py:171 vs :122), the published windowed-attention claim (sweeps
s at fixed w=8 where the statistic reads only [i-16, i] and is constant in s BY
CONSTRUCTION), and the pivot selector itself (ranks on key.norm, which the
operator L2-NORMALIZES AWAY -- a 7.5x key rescale moves A by **2.98e-08**, and
content vs uniform-random pivots differ by 3.1%).

## SUCCESSORS - required by the standing rule; both fellows delivered

**M2' (Foreman):** stop claiming flatness; claim the **trade curve**. Measure
sign_flip_rate jointly with ||A^2|| across the lambda family already coded
(tgate, tgatex, sgate). **Pre-registered kill: if rate * ||A^2||^(1/2) is not
approximately invariant across the three arms, the lemma is wrong.**

**The designated-token repair (Cameron):** M2's c-not-in-P, M2's
content-selection claim and M4's kill are ONE bug -- the quantity under test is
drawn from the mechanism under test. **Repair: the designated token must come
from the TASK, never from the selector.** That is why **M3 is not the next item,
it is the ONLY instrument that has one.**

## CORRECTIONS TO NUMBERS PREVIOUSLY PUBLISHED HERE

- **--collect-only is NOT 847.** Measured **1,036** at 12:10 and **1,041** at
  12:47 -- **the count moves live.** Every fixed collection number in every
  document is wrong the moment it is written.
- **129.7 / 257.2 A100-h are NOT sizing.py's output.** gpu_hours gives 41.4/46.8
  for the CONTROL; 129.7/257.2 are control x the measured 3.13x/5.49x ratio and
  are **the operator arm ALONE**. **The matched pair is 171.1 / 304.0.**
- Colab memory was stale: **2.72 GiB against 13.50 GiB**, not 2.45/14.5.
- pivot_unsigned HAD run once (iteration 3, n=128, both 0.0000) -- never in the
  M2 journal, never at protocol scale.

## FREE COMPUTE - the answer to what ships

- **Largest free, NO resume: ~37.8M params** (one 12 h Kaggle session, MFU 0.30);
  25.2M at MFU 0.15.
- **With the resume patch (Chase built it, verified bit-exact): ~100M** in 4 weeks
  of Kaggle's 30 h/week, ~177M in 12 weeks.
- **300M gate: NOT REACHABLE FREE.** 171.1 A100-h matched = **1,095 T4-hours =
  37 weeks** of the entire free quota. **Paid: $46-50 on vast.ai**, $204 RunPod.
- **Memory was NEVER binding** -- a 300M model FITS a free T4 (ceiling 308M, 737M
  with checkpointing). **Time is what stops it.**
- **What ships: the notebook's 25.7M default**, 7.0x the 3.65M ceiling this
  project has ever trained, inside one free session.
