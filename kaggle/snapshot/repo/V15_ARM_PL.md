# V15 MERCURY-2 — ARM PL, built and bound

Node `MERCURY-2` of the CEQ v15.2 composition round. Builds the arm that has
been blocked behind L-LEAN since `it.0`; the block lifted when
`lean/CEQ/V15Fork.lean`'s identity theorems went green.

The operator is `V15_JUPITER2_FORK.md` §8.3 and is **not redesigned here**:

```
l_ij  =  q_ij  -  C_j  +  s_j          j = 0..i,   C = scan(g)
O_i   =  sum_{j<=i} softmax_j(l_i.) * V_j
```

`g` and `s` are two independent per-position scalar heads; position `0` is a
BOS slot. The query-side scan term is **absent by settlement**, not by
omission: under a causal softmax the `C_i` half is annihilated identically
(§4, `max |out − out with C_i deleted| = 0`), so §8.4 item 3 deletes it and
`ceq/arm_pl.py` implements `-C_j`. `VARIANT = "key_only"` records that in every
manifest, so a cell measured under the query-side form cannot be filed under
this one.

**Nothing was trained.** No optimizer, no gradient step, no R1, no R2. Every
number below is a float64 identity residual or a bitwise comparison. Training
is the next node's and it is gated on these binds.

Files written: `ceq/arm_pl.py` (new), `tests/arm_pl/test_arm_pl.py` (new),
`tests/arm_pl/conftest.py` (new), this file. No existing module was touched. No
git command that writes was run.

---

## VERDICT TABLE

| bind | contract bar | measured | verdict |
|---|---|---|---|
| **1 — parity** | bitwise | `torch.equal` **True**, against two independently written references | GREEN |
| **2 — label** | `<= 1e-6` | `6.6613381477509392e-16` | GREEN |
| **3 — DAG resolvent** | `1.1e-16` `[INHERITED]` | `2.78e-17` absolute (light gates); `3.55e-15` absolute / `2.16e-16` relative (heavy gates, resolvent entries to `16.47`) | GREEN, re-measured |
| **planted negatives** | all four must fail at `O(1)` | `0.9749`, `0.9165`, `1`, `0.4845` — four for four, matching the published digits exactly | all four fired |

`tests/arm_pl`: **17 passed**. `tests/loop`: **15 failed / 510 passed**, against
a pre-existing baseline of **15 failed / 509 passed** measured at `deee6c4`
before any file here was written. The single added case is
`test_no_module_writes_a_file_at_import.py::…[ceq/arm_pl.py]`, the repo-wide
module sweep picking up the new file, and it passes. **Zero new failures.**

---

## 1 — BIND 1, AND WHAT IT DOES NOT SAY

### The honest wording

> **This bind's information content is: the modification enters only through
> the key logit, additively, and vanishes at zero.**

It is **not** "bitwise standard attention" unqualified. That is what §S-M's
original clause claimed, and it is what `CEQ.V15.gate_zero_not_stochastic`
refuted by showing the old operator's rows sum to `i + 1`. `MISTAKES.md` V-24's
closing paragraph is binding on this and is followed to the letter: the
replacement's parity bind sits at *"both auxiliary heads zero"*, its rejection
region is non-empty but **narrower** than the refuted original's, which asserted
parity at *the gate's own identity point*. The two binds sit at opposite ends of
one degeneration — (P) is `a ≡ 1` and (L) needs `a ≠ 1` (§8.4 item 5) — so there
is no `θ` at which both hold, and there was never going to be.

### The measurement

```
  (P) operator bitwise == lm.Attention(softmax_x)      True
  (P) readout  bitwise == ref @ V                      True
  (P) ArmPL._operator at zero_heads(), bitwise         True
  (P) diagonal=-1 bitwise == bench._softmax_operator   True
  rejection region: g != 0 moves the operator by       0.31083070671207125
  rejection region: s != 0 moves the operator by       0.3644083108139286
```

`torch.equal`, not `allclose`. Four rows, and each is a separate claim.

### The reference, and why it is not `ceq/bench.py` alone

The brief named `ceq/bench.py`'s softmax operator or `ceq/attention.py`. Neither
is usable as the *primary* reference, and the reason is a measurement rather
than a preference:

- `bench._causal_mask_pair` is `tril(-1)` — **strictly** causal, diagonal
  EXCLUDED. Row 0 is then empty and `A_ii = 0`. ARM PL needs `P_ii = 1`, so a
  strictly-causal reference cannot carry the (L) bind's convention.
- `ceq/attention.py` is the signed row-L1 operator, not a softmax at all.

Using a strictly-causal reference for (P) and an inclusive-causal operator for
(L) would make the two binds statements about two different objects — precisely
the V-24 *shared carrier* defect the round is guarding against. So the primary
reference is `ceq/lm.py`'s `Attention("softmax_x").operator`, the repo's own
INCLUSIVE-causal armed control, whose header states at `lm.py:39` that
`SMX_TAU=1, SMX_RHO=1, SMX_HOPS=1, SMX_ID=False` **is** exactly standard causal
attention. `bench._softmax_operator` is then used as a **second** reference at
the strict convention, so the bind is run against two independently written
implementations rather than one.

### The bind is also run on the object the harness would hold

`test_bind1_holds_for_the_drop_in_module_and_not_only_the_function` asserts the
same equality on `ArmPL._operator` after `zero_heads()`. A parity bind on a free
function the shipped module does not dispatch through is a reading of a
non-shipped operator — the defect
`tests/loop/test_m3_harness_operator_is_shipped.py` exists for.

### Class closure

V-24's first test, evaluated at random `θ` and not only at `θ₀`:

```
  max |row sum - 1| over 3 random (g, s)               < 1e-15
  min entry over 3 random (g, s)                       > 0
```

The family never leaves the softmax class. That is what separates this bind from
the two-branch escape V-24 rules vacuous, whose object is outside the class at
every `λ ≠ 0`.

---

## 2 — BIND 2, THE LABEL

Oracle setting: `q ≡ 0`; `g_0 = 0`, `g_j = log a_j`; `s_0 = 0`,
`s_j = log(1 − a_j)`; `V_0 = 0`, `V_j = b_j/(1 − a_j)`. Draw `s = 8`, seed `15`,
gates uniform on `(0.15, 0.85)` — the fork probe's own draw, so the numbers are
comparable to the published ones.

```
  (L) max |O_i - y_i|   [contract bar 1e-6]            6.6613381477509392e-16
  (L) max |Z_i - 1|     THE TELESCOPING                4.4408920985006262e-16
  (L) min entry of the attention matrix                0.0006633673588462318
  (L) max |row sum - 1|                                3.3306690738754696e-16
  (L) max |y_i| (the scale the residual is read against) 1.9843520536084989
  (L) same A, same V, numpy matmul instead of torch    4.4408920985006262e-16
```

**The reported number is `6.66e-16` and not the `2.2e-16` the fork published.**
Both are `2.2e-16`-class — 3 ulp against 1 ulp of a quantity whose own scale is
`1.98` — and the difference is matmul reduction order, measured rather than
assumed: the *same* attention matrix and the *same* values, read out through
numpy instead of torch, give `4.44e-16`. The construction is unchanged; the
BLAS is not. Against the contract's bar of `1e-6` the margin is nine orders.

`min entry = 6.6e-4 > 0` and `max |row sum − 1| = 3.33e-16` say the row is a
strictly positive probability vector, hence a genuine softmax row — the object
really is inside the class, which is `Asink_nonneg` and `Asink_row_sum` as
measurements rather than as theorems.

---

## 3 — BIND 3, THE DAG RESOLVENT, RE-MEASURED

`CEQ_V15_CONTRACT.md`:98 carries `[RUN: matches brute-force path sums to
1.1e-16]`. `V15_CONTRACT_ARITHMETIC_AUDIT.md`:273-276 marks every such PART I
figure `INHERITED` under L-TIME and requires the citing iteration to re-measure
it. Re-measured:

```
  light gates (0.05,0.45) p=0.6: max |R - brute|       2.7755575615628914e-17
    max entry of R / relative residual / edges         1 / 2.78e-17 / 19
  heavy gates (0.30,0.90) p=1.0: max |R - brute|       3.5527136788005009e-15
    max entry of R / relative residual / edges         16.468587948382389 / 2.1572667249528595e-16 / 36
```

**Two regimes, deliberately, and the second is the one that carries the
evidence.** At light gates the resolvent's entries barely leave `1`, so an
absolute agreement of `1e-17` is cheap — it is a residual on a quantity that is
nearly the identity. The heavy regime drives the entries to `16.47`, and the
relative residual there is `2.16e-16`, one ulp. The inherited `1.1e-16` is an
absolute figure with no stated scale; it reproduces as `2.78e-17` at the light
regime and as `2.16e-16` **relative** at the heavy one.

**The two routes share no code.** `dag_resolvent` is a dense LU solve, which
knows nothing about paths. `brute_force_path_sums` enumerates every increasing
vertex sequence `j < … < i` and multiplies its edge gates — the definition,
counted. Comparing them is a check and not an identity restated twice
(`MISTAKES.md` V-3).

---

## 4 — THE PLANTED NEGATIVES

Not an extra. `MISTAKES.md` V-24 exists because a bind with an empty rejection
region is passed by the answer key, and its check clause requires that
*"deliberate mutilations of the mechanism make it fail"*.

```
    drop_key_bias            max|out - y| = 0.97494590151405114   published 0.97494590151405114
    drop_value_rescale       max|out - y| = 0.9165274652308163    published 0.9165274652308163
    drop_bos_sink            max|out - y| = 1                     published 1
    half_key_bias            max|out - y| = 0.48449311856985267   published 0.48449311856985267
```

**Four for four, and every printed digit matches the published figure.** The
draw is reproduced exactly (`np.random.default_rng(15)`, gates on `(0.15,0.85)`,
drives standard normal), so this is a reproduction of §4's plants rather than a
new draw that happens to land nearby. No plant failed to fire, so there is no
finding to report against them.

Each mutilation removes exactly one part of the construction and nothing else:
`s ≡ 0`; `V_j = b_j`; `V_0 = 1`; `s_j = log(1 − a_j)/2`. The contrast with V-24's
own vacuous case — three substitutions including the label itself, three passes
— is the operational difference between an informative identity bind and a
decorative one.

---

## 5 — THE DIAGNOSTIC COLUMN, A PREDICTED FAILURE MODE FILED BEFORE THE ARM RAN

§8.4 cost 6: the value dynamic range is `1/(1 − a)`, reaching `1e6` at
`a = 1 − 1e-6`. `label_cell` emits `v_max` against `dyn_range_bound` as a
per-cell column of the manifest, and
`test_the_dynamic_range_column_is_populated_and_grows_as_a_goes_to_one` fails if
the column is absent, non-finite, or flat.

Constant gates, `b ≡ 1`, `s = 8`:

```
  a_max      v_max          1/(1-a_max)    max|y_i|       v_max/max|y| (L) residual
  0.5        2              2              1.9921875      1.00392      0
  0.9        10             10             5.6953279      1.75583      1.77636e-15
  0.99       100            100            7.7255306      12.9441      2.66454e-15
  0.999999   1000000        1000000        7.999972       125000       5.32907e-15
```

Three readings.

1. **The column tracks the bound exactly.** `v_max = 1/(1 − a_max)` to the digit
   at every row, which is the construction paying the `s → ∞` optimum. §5's
   over-payment ratios reproduce: `1.004`, `1.756`, `12.94`, `1.25e5`.
2. **The label bind degrades in the predicted direction.** The residual runs
   `0 → 1.78e-15 → 2.66e-15 → 5.33e-15` as `a → 1`: catastrophic cancellation
   between a BOS slot carrying nearly all the softmax mass and values of size
   `1e6`. Still nine orders inside the bar at `a = 1 − 1e-6`, so the identity
   survives the whole range — but the *direction* is the one §8.4 item 6
   predicted, and it is now measured rather than argued.
3. **The failure mode is not yet reached.** Expressibility is intact at `1e6`
   value scale. VENUS's prediction is about *learnability*, and nothing here
   tests it.

---

## 6 — THE IDENTITY MANIFEST

Every cell carries one. `scale/identity_manifest.py` is **not edited** — it is
another node's file and every published hash depends on `CONFIG_FIELDS` — so the
base manifest is computed by its own `manifest()` and an ARM-PL block is hashed
alongside and folded into the combined hash.

```
  hash        c397733af9a20a80be5d5a667c2d50fd51ff0b24d7b15b66a134ce23caf98c2f
  pl block    {'variant': 'key_only', 'g_setting': 'log a_j, g_0 = 0',
               's_setting': 'log(1 - a_j), s_0 = 0', 'bos_value': 0.0,
               'a_max': 0.7210719779352402, 'v_max': 3.954017755450676,
               'dyn_range_bound': 3.585154308260309}
  config      {'cell': 'arm_pl:none', 'kind': 'arm_pl', 'task': 'chain_label',
               's': 8, 'd': 1, 'd_model': 1, 'steps': 0, 'seed': 15,
               'device': 'cpu', 'torch_version': '2.5.1+cu121'}
  absent      ['k_piv', 'beta', 't_max', 'n_neumann', 'n_train', 'n_eval']
```

- `device` is a declared `CONFIG_FIELD` and this cell **carries** it. The
  manifest module's own note records that every shipped weight record omits it;
  these cells do not.
- `absent` names the six declared fields this cell does not carry, and every one
  of them is a training-harness field. `steps = 0` is carried and is literally
  true: this node runs no gradient step.
- The head SETTING is hashed as the `rng` component, because `params` is the
  actual `(g, s, V)` triple — for an untrained arm the head setting *is* the
  parameter vector.
- The hash moves under every mutilation, which is what makes it an identity:

```
    drop_key_bias        8a0e16552da7875f...
    drop_value_rescale   9a973fb782d96dcf...
    drop_bos_sink        7d37b0fe7e62448e...
    half_key_bias        1fd658659f97154f...
```

---

## 7 — TDD: RED, THEN GREEN

### RED — tests written first, module absent

```
$ python -m pytest tests/arm_pl -q -p no:cacheprovider
=================================== ERRORS ====================================
________________ ERROR collecting tests/arm_pl/test_arm_pl.py _________________
ImportError while importing test module 'C:\Users\seal\Desktop\New folder (32)\tests\arm_pl\test_arm_pl.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\arm_pl\test_arm_pl.py:47: in <module>
    from ceq import arm_pl, bench, lm
E   ImportError: cannot import name 'arm_pl' from 'ceq' (C:\Users\seal\Desktop\New folder (32)\ceq\__init__.py)
=========================== short test summary info ===========================
ERROR tests/arm_pl/test_arm_pl.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 2.56s
```

### GREEN — `ceq/arm_pl.py` written

```
$ python -m pytest tests/arm_pl -q -p no:cacheprovider
.................                                                        [100%]
17 passed in 2.13s

$ python -m pytest tests/arm_pl -v --no-header -p no:cacheprovider
tests/arm_pl/test_arm_pl.py::test_bind1_parity_is_bitwise_against_the_inclusive_causal_control PASSED
tests/arm_pl/test_arm_pl.py::test_bind1_parity_is_bitwise_against_bench_at_the_strict_convention PASSED
tests/arm_pl/test_arm_pl.py::test_bind1_holds_for_the_drop_in_module_and_not_only_the_function PASSED
tests/arm_pl/test_arm_pl.py::test_bind1_has_an_occupied_rejection_region PASSED
tests/arm_pl/test_arm_pl.py::test_the_family_never_leaves_the_softmax_class PASSED
tests/arm_pl/test_arm_pl.py::test_bind2_the_oracle_setting_reproduces_the_chain_label PASSED
tests/arm_pl/test_arm_pl.py::test_bind2_the_normalizer_telescopes_to_one PASSED
tests/arm_pl/test_arm_pl.py::test_bind3_the_dag_resolvent_matches_brute_force_path_sums[0.05-0.45-0.6] PASSED
tests/arm_pl/test_arm_pl.py::test_bind3_the_dag_resolvent_matches_brute_force_path_sums[0.3-0.9-1.0] PASSED
tests/arm_pl/test_arm_pl.py::test_every_planted_negative_fires[drop_key_bias-0.9749] PASSED
tests/arm_pl/test_arm_pl.py::test_every_planted_negative_fires[drop_value_rescale-0.9165] PASSED
tests/arm_pl/test_arm_pl.py::test_every_planted_negative_fires[drop_bos_sink-1.0] PASSED
tests/arm_pl/test_arm_pl.py::test_every_planted_negative_fires[half_key_bias-0.4845] PASSED
tests/arm_pl/test_arm_pl.py::test_the_mutation_list_is_the_one_the_fork_settled PASSED
tests/arm_pl/test_arm_pl.py::test_the_dynamic_range_column_is_populated_and_grows_as_a_goes_to_one PASSED
tests/arm_pl/test_arm_pl.py::test_the_manifest_records_the_variant_the_heads_the_bos_and_the_column PASSED
tests/arm_pl/test_arm_pl.py::test_the_manifest_hash_moves_when_a_planted_negative_moves_the_cell PASSED
============================= 17 passed in 2.49s ==============================
```

### The standing suite

```
$ python -m pytest tests/loop -q --no-header -p no:cacheprovider 2>&1 | tail -5
FAILED tests/loop/test_every_boundary_node_can_propagate.py::test_no_corpus_instance_has_a_boundary_node_that_cannot_propagate[reproduce-n1024-d4-t0.95]
FAILED tests/loop/test_manifest_refuses_an_absence_it_has_not_earned.py::test_no_weight_record_omits_a_declared_identity_field
FAILED tests/loop/test_the_bar_control_is_scored_out_of_sample.py::test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on
15 failed, 510 passed, 3 warnings in 31.62s
```

**The standing count in the brief is stale.** The baseline was measured at
`deee6c4` *before* any file here was written and read **15 failed / 509
passed**, not `15 failed / 507 passed`. After this node it reads
**15 failed / 510 passed**; the one added case is

```
tests/loop/test_no_module_writes_a_file_at_import.py::test_module_has_no_import_time_write_or_argv_read[ceq/arm_pl.py]
```

— that test parametrizes over every module in the repo, so a new module adds a
case by construction. It passes. The 15 failures are the same 15, by name. **No
failure here is new.**

---

## 8 — WHAT WAS BUILT

`ceq/arm_pl.py`, and the whole of it:

| name | what it is |
|---|---|
| `scan`, `key_bias` | `C = cumsum(g)`; the key-side bias `-C_j + s_j` |
| `operator` | the `[…,S,S]` causal softmax row; `diagonal` selects the mask convention |
| `readout`, `normalizer` | `A @ V`; `Z_i = exp(C_i) Σ exp(s_j − C_j)`, the telescoping |
| `oracle_heads`, `chain_label` | the (L) setting; the label by its own recurrence |
| `draw`, `constant_gate_draw` | the fork probe's draw; §5's constant-gate row |
| `mutate`, `MUTATIONS` | the four planted negatives, one part removed each |
| `label_cell` | one measured cell: residual, dynamic-range column, manifest |
| `draw_dag`, `dag_resolvent`, `brute_force_path_sums` | bind 3's two independent routes |
| `cell_manifest`, `PL_FIELDS` | `identity_manifest.manifest` plus the ARM-PL block |
| `ArmPL` | the `nn.Module` drop-in for `scale/m3_capability.py`'s harness |

`ArmPL` matches `m3_capability.Arm`'s shape — `wq`/`wk`, one shared-width MLP, a
scalar readout, `forward(x) -> [n]` reading position `s − 1`. Two things differ
and they are the whole architecture: the two per-position scalar heads whose
output enters the KEY logit additively, and the value path the (L) setting
rescales. `_operator` takes `(q, k, g, s)` rather than `(q, k)`; the extra two
arguments *are* the arm. It carries no optimizer and this node constructs no
gradient.

---

## LIMITS

- Everything above is exact arithmetic and float64 identity at `s = 8`, one
  draw, seed `15`, gates in `(0.15, 0.85)` — plus the constant-gate rows of §5.
  Nothing here is evidence that a trained arm finds any of these settings. The
  binds are expressibility, not learnability, exactly as the contract's binds
  are.
- The (L) setting has `q ≡ 0`, so at the oracle setting the head is not doing
  content addressing. Inherent to the bind's design and shared with §S-M's own
  oracle-gate bind; not introduced here.
- Bind 1's primary reference is `ceq/lm.py`'s `softmax_x` control and not
  `ceq/bench.py`'s, because bench's mask is `tril(-1)` and excludes the diagonal
  the label needs. `bench._softmax_operator` is still used, at the strict
  convention, so two references carry the bind — but neither is a
  *fresh-eyes* reimplementation written for this node, and both live in this
  repo.
- Bind 2's `6.66e-16` is 3 ulp where the fork published 1 ulp. The gap is matmul
  reduction order and is demonstrated as such (numpy read-out of the identical
  matrix gives `4.44e-16`); it is not a difference in the construction.
- Bind 3's inherited `1.1e-16` had no stated scale, so the comparison is not
  exact. It is re-measured in two regimes and both are reported rather than the
  flattering one alone.
- The dynamic-range column measures value scale and the (L) residual. It does
  **not** measure the failure mode §8.4 item 6 predicts, which is a *learning*
  failure; that reading belongs to the training node and is still owed.
- §6 of the fork — the signed variant — is untouched here and is still open.
- `ceq` importing `scale` is a new dependency direction in this repo. The
  alternative was a private clone of `CONFIG_FIELDS`, and two copies of an
  identity rule is the defect that rule exists to catch.
- `ArmPL` is written to the harness's interface and is **not exercised by the
  harness** in this node. Only its operator and its forward shape are bound.
