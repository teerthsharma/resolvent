# HEALTH INSPECTOR — CEQ v20 ROUND 15, it.2

Office of the log. Authority over **binding only**: whether a claim is tied to a
RED that was RED first, and whether a published number reproduces. Never whether
a claim is correct. No stance, no findings, no appeal.

Same rules as `V20_R15_IT1_INSPECTOR.md` (26 audited, 7 struck), including the
discriminator established there: **a collection-error RED binds the SUITE, not a
proposition — it binds exactly those claims some node actually asserts, and
nothing else.**

Nurses: 8 (5 haiku re-run/diff, 3 sonnet mutation). No git write. Nothing
touched Kaggle.

---

## CHECK 1 — EVERY TEST CLAIMED GREEN, RE-RUN

| claim | claimed | re-run | verdict |
|---|---|---|---|
| `tests/jupiter/` suite | 92 → 101 passed | **101 passed in 44.23s** | **clean** |
| `tests/jupiter/test_v20_r15_it2_ldom_census.py` | 9 nodes, 9 passed | **9 collected, 9 passed** | **clean** |
| `tests/saturn/test_v20_r15_wings_distinct.py` | 5 passed | **5 passed in 1.82s** | **clean** |
| `tests/saturn/` suite | 59 passed / 7 failed, 6 pre-existing | **59 passed, 7 failed in 6.20s** | **clean** |
| `tests/mars_v20/` directory | 16 RED / 14 GREEN | **16 failed, 14 passed in 5.65s** | **clean** |

SATURN's "6 of 7 pre-existing" checks out by inspection: the seventh failure is
`test_v20_r15_wing_rubric.py::test_every_annex_run_instance_has_a_producer_in_the_tree`,
failing on exactly `{'M9-F1 Cantelli/Boole cutoff': 'd=65', 'M2 gate-landscape
theta': '1.3e-3'}` — his own K6 standing RED, the two survivors he declares.
The claim is self-consistent and reproduces.

MARS's directory reproduces to the node. The 14 GREEN are all and only the
`test_control_*` nodes plus three manifest/planted-positive nodes.

---

## CHECK 3 — THE THREE-WAY COLLISION, AS A BINDING QUESTION ONLY

Whether these strikes move the round's verdict is the coordinator's call. This
office rules on two things only: was each strike RED first, and does each number
reproduce.

### STRIKE 5 — the saturated gate — **clean**

RED at log `11726`, finding at `11729`. **RED first.**

- `ceq/arm_smprime.py:109-113` — `def magnitude(u)` returning
  `torch.clamp(u, 0.0, 1.0)`. MARS cites `:109`, the `def` line; the `clamp`
  itself is `:113`. Same function, accepted.
- **99.17% reproduces.** Analytic `(116.006073 − 1.0)/(116.006073 − 0.0318116)
  = 0.991652`. Independent 1e6-sample simulation at seed 0: `0.991671`. Both
  round to `99.17%` at 2dp.
- **The disclosure MARS attributes to SATURN exists.**
  `V20_R15_IT2_SATURN.md:241-244`: W1's trained gate vectors "are likewise
  unjournalled, so W1's readings use a gate drawn from the journalled range
  `[0.0318116, 116.006]` rather than the trained vectors themselves". The
  second defect is disclosed in the same paragraph, `:244-246`: `seconds_to_floor`
  "counts cumulative `secs` over journal order to the first crossing cell …
  summed in file order, not a wall-clock schedule."

  **Precision note, not a strike.** What `Limits` discloses is the *substitution*
  (drawn range in place of trained vectors), not the *saturation* (99.17% of the
  draw pinned at exactly `1.0`). MARS claims only the former — "SATURN discloses
  both defects below in his own `Limits` paragraph" — and states the
  quantification is his, "quantified here for the first time". The attribution
  is accurate as written.

### STRIKE 6 — the row-order artifact — **clean**

RED at log `11727`, finding at `11730`. **RED first.**

Both orderings recomputed independently against `results/v17k_r4_retake.jsonl`,
`FLOOR_1 = 0.7071067811865476`, by re-implementing `s2f`
(`tests/mars_v20/test_saturn_it2_distinctness_gate_and_cost.py:93-99`) from
scratch:

| ordering | W3 `arm_pl` | W1 `arm_smprime` | ratio |
|---|---|---|---|
| file order | `1.884` (row 0) | `47.048` (row 2) | `24.972` → **25.0×** |
| best-first | `1.759` | `14.852` | `8.443` → **8.4×** |

Cross-checked against the second, independent implementation at
`tests/saturn/test_v20_r15_wing_rubric.py:433-448`, file order:
`{'arm_pl': 1.884, 'arm_smprime': 47.048}` — matches. Both figures reproduce.

### STRIKE 7 — the unnamed cap — **clean**

RED at log `11733`/`11734`, finding at `11736`. **RED first.**

Recomputed from `results/v17k_r4_retake.jsonl` on the W3 = `arm_pl` bind cells
(`kind == "arm_pl"`, the row carrying `a_hat_min`/`eval_nrmse`):

| seed | `a_hat_max` | `g = ln a_hat_max` | MARS | `eval_nrmse` |
|---|---|---|---|---|
| 0 | 1.4104527 | 0.3439 | 0.3439 | 0.6446726 |
| 1 | 1.2868506 | 0.2522 | 0.2522 | 0.6445175 |
| 2 | 12.7675161 | 2.5469 | 2.5469 | 1.1522795 |
| 3 | 49.6605225 | 3.9052 | 3.9052 | 1.1133392 |
| 4 | 1.4536346 | 0.3741 | 0.3741 | 0.6337391 |
| 5 | 1.5051768 | 0.4089 | 0.4089 | 0.6419986 |
| 6 | 1.1029453 | 0.0980 | 0.0980 | 0.6621282 |
| 7 | 116.0060730 | **4.7536** | 4.7536 | 1.1489268 |

**8 of 8 exceed `1.0`**, so `g > 0` on every trained cell; max `g = 4.7536` at
seed 7. All eight logarithms reproduce to 4dp.

Partition: five cells at `a_hat_max <= 2.0` (seeds 0/1/4/5/6) top out at
**`0.662128`**; three above (seeds 2/3/7) bottom out at **`1.113339`**. Exactly
5/3, zero overlap. Both endpoints reproduce to 6dp.

**Precision note, not a strike.** The cap boundary is `1.0` and every cell
exceeds it, so `1.0` does not partition anything; the threshold that partitions
is `2.0`. MARS's assertion text says exactly that — "5 cells at `a_hat_max<=2.0`"
— and is precise. His surrounding prose calls it "the cap boundary … partitions",
which conflates the cap with the partitioning threshold. The numbers are right;
the sentence is loose.

**Nurse correction recorded.** The first nurse sent at STRIKE 7 mapped W3 to
`kind == "softmax"` and returned "CLAIM 1 IS FALSE, 0 of 8". `softmax` reads
`a_hat_max = 1.0` on all eight cells. W3 is `arm_pl`
(`V20_R15_IT2_SATURN.md:32`, `ceq/arm_pl.py:99`). The refutation was the nurse's
arm-mapping error and is withdrawn by this office; the strike stands clean.

---

## CHECK 4 — THE COORDINATOR

Not exempt. Both tables recomputed by four methods: `scipy.stats.beta.ppf`;
`scipy.stats.binomtest(...).proportion_ci(method='exact')`; a from-scratch
bisection on the exact binomial tail using `math.comb` with `Fraction` exact
rational arithmetic; and the same bisection in floats. All four agree to ≥ 6dp.

**Clopper–Pearson table (`V20_R15_JOURNAL.md:266-271`) — clean.**

| k/8 | claimed | two-sided 95% (α_lo = 0.025) | one-sided 95% (α_lo = 0.05) |
|---|---|---|---|
| 5 | `0.2449` | **0.244863** ✓ | 0.289241 ✗ |
| 1 | `0.0032` | **0.003160** ✓ | 0.006391 ✗ |
| 0 | `0.0` | **0.000000** ✓ | 0.000000 ✓ |

The journal declares "two-sided 95% Clopper–Pearson" at `:266` and that is the
convention that reproduces. The one-sided convention does not. Correctly labelled.

**Sizing claim (`:275-278`) — clean, with the interpretation named.**

`8/8 → 0.630583`, `7/8 → 0.473490`, `6/8 → 0.349144`. Only `8/8` clears `0.5`,
as claimed. All three round to the published figures.

`N = 65`: the first `N` whose CP-lower clears `0.5` at rate `0.625` is `N = 65`
**under `k = round(0.625·N)`** — `N=64, k=40, CP-lower 0.495076` (short);
`N=65, k=41, CP-lower 0.501999` (clears). The journal's own parenthetical reads
`(41/65, CP-lower 0.5020)`, which names `k=41` explicitly, so the interpretation
is published, not inferred. **Recorded for the record:** `k = ceil` gives `N=58`
and `k = floor` gives `N=72`; the claim is interpretation-sensitive and the
journal pins the interpretation. Clean.

---

## CHECK 2 — EVERY FINDING HAS A MATCHING RED BEFORE IT

Walked in log order, `house-events.jsonl` lines `11700`–`11737` (the it.2 tail;
line `11700` is this office's it.1 `done`).

### The discriminator, applied to JUPITER's L-DOM RED

The it.2 RED is again an `ImportError`. Re-applied by hand — the definition
`scripts/v20_m14_cheeger.py:429 def domain_census_facts()` renamed, the import at
`tests/jupiter/test_v20_r15_it2_ldom_census.py:31` left alone:

```
E   ImportError: cannot import name 'domain_census_facts' from 'scripts.v20_m14_cheeger'
Interrupted: 1 error during collection
1 error in 0.59s
```

**It is a collection error: `1 error`, zero tests run.** By the it.1 rule it
binds the SUITE, and within the suite exactly those claims some node actually
asserts. The nine nodes assert two propositions and no third:

- `test_ldom_census_bed_m_...`, `..._the_ternary_object_...`,
  `..._bed_k_is_strictly_lower_triangular_and_nilpotent_...`,
  `..._verdict_zero_registered_beds_...` — **the L-DOM census is BOUND.**
- `test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided` (both sides plus
  `crude / exact == approx(1.0, abs=TOL)`) and
  `test_the_repo_crude_phi_is_the_planted_rational_one_over_twentyseven`
  — **`crude == exact` is BOUND.** This closes it.1's C20 strike, which struck
  the one-sided form for admitting `crude = 10 x exact`.
- **Nothing asserts the merge.** See the strike below.

### The count on the RED — struck

The log event (`11703`) reads `ImportError domain_census_facts (8 nodes
uncollectable)`; the report says all 9 were RED. The file defines **9** test
functions, and a module-level `ImportError` takes the whole module — pytest
reports `1 error during collection` and **no per-node result at all**. Neither
`8` nor a nine-way RED is what the tool emits. **The RED is real and binds the
suite; the count is struck**, the same class as it.1's `m14_planted_negative_A`
count.

### JUPITER's N=1-primitive merge verdict — struck, UNBOUND

Logged as a finding at `11720`. A grep for `primitive`, `isomorph`, `cumprod` and
`merge` across `tests/jupiter/` returns **zero files**; no node in any of the nine
`tests/jupiter/` modules names the merge, and none of the nine L-DOM nodes
asserts it. The `ImportError` RED cannot reach it — a suite-level RED binds what
the suite asserts, and the suite does not assert this.

Noted as fact, not as relief: the only test in the tree that asserts JUPITER's
algebra is **MARS's** control,
`tests/mars_v20/test_jupiter_it2_merge_hides_the_upper_cap.py::test_control_the_isomorphism_is_exact_where_both_wings_are_defined`
(GREEN). An adversary's control is not the author's bind, and in any case it
asserts only `exp . cumsum . log == cumprod`, not the `N = 1` verdict over
W1/W2/W3.

### JUPITER's planted negative for the two-sided node — struck

`test_planted_negative_the_two_sided_node_fires_when_crude_is_inflated` does not
run the node it names. It re-declares the assertion against a local:

```python
mutated = crude * 10.0
assert mutated >= exact - TOL              # the ONE-SIDED node still passes
with pytest.raises(AssertionError):
    assert mutated <= exact + TOL, "upper side"   # the two-sided node fires
```

`test_the_repo_crude_phi_EQUALS_the_exact_minimum_two_sided` is never invoked.
**Delete the upper-side assertion from the real node and this planted negative
still passes** — it guards a copy of the text it is supposed to guard. The
proposition stays BOUND on the real node; the claim that *the node fires* is
struck.

### JUPITER's seeded permutation — clean, and it repairs C7

`numpy.random.default_rng(20152).permutation(n)`, applied as `W[np.ix_(p, p)]`,
is named at `V20_R15_IT2_JUPITER.md:293-294` and lives in
`test_planted_negative_the_sweep_path_is_not_hardwired_to_the_exact_path`. Seed
and generator are both specified, non-identity is asserted inside the node, and
re-applying it to the production `W` in `scripts/v20_m14_cheeger.py` leaves
`9 passed` — which is what the node claims, since it is an **invariance** check,
not a fire-mutation. It.1's C7 struck "a scrambled ordering" for naming no
permutation. That defect is repaired.

**Correction to the brief this office was given.** The seeded permutation is not
attached to the two-sided `crude == exact` node. JUPITER attaches it to the
sweep-path node and attaches `crude * 10.0` to the two-sided one (`:287-298`).
His document is precise here; the conflation is not his.

### JUPITER's six Lean withdrawals — clean

A withdrawal is not a finding and needs no RED, but it must be written down. It
is: `V20_R15_IT2_JUPITER.md:325-333`, a table with a verdict per row. Six
distinct items withdrawn — **#18, #19, #17', #20, #21, #23** — with #22
reconciled by name to `lean/CEQ/V15Source.lean:128 two_sources_recovered` rather
than withdrawn. Six, enumerated, in the report.

### SATURN's it.2 log — struck, as a log claim only

A grep for Saturn as an `agent` field over the it.2 tail returns **0**. SATURN
filed **one** event at it.2, line `11719`, and it carries no `t` field at all —
no `t:"test"`, no `status:"red"`, no `t:"finding"`, no `t:"done"`. It is a single
composite record filed at close, holding `red_at_corner`,
`green_min_over_8_trained_seeds`, the verdict, `B_K5`, `C_K6` and the suite line
together in one atom.

**What is struck is the ordering, not the content.** Nothing in the log
establishes that his RED was RED *first*, because the RED and the verdict arrive
in the same event. His round is not walkable in log order.

**What is not struck:** the underlying claims. The RED is real, quoted verbatim
at `V20_R15_IT2_SATURN.md:66-76`, and is a mutation-RED of the proper shape —
the bind's own assertion re-run with MARS's corner substituted for the trained
settings, firing at `{'W1/W2': 1.11e-16, 'W1/W3': 1.11e-16, 'W2/W3': 0.0}`. Two
standing planted positives hold the instrument
(`test_the_measure_calls_an_arm_against_itself_the_same`,
`test_the_measure_calls_marss_corner_the_same`), and the suite reproduces at
5 passed. **The distinctness verdict is BOUND on the tests.**

- **K5 withdrawal — clean.** Written down, not merely mentioned:
  `V20_R15_IT2_SATURN.md:127`, `## B — K5: WITHDRAWN, refuted by its own node`,
  with the refuting `[RUN]` beneath it.
- **K6 halving 4 to 2 — clean.** `:203-204`, survivors `d=65` and `1.3e-3`,
  withdrawn `d=20 at the same` and `0.492 vs KL 0.519` — the two this office
  struck at it.1 (C25, C26). The standing RED reproduces naming exactly those
  two survivors.
- **V-7b — clean.** Written at `:206-224` as a `MISTAKES.md` entry with
  mechanism, detection and repair shape.

### MARS — clean throughout, RED before every finding

| finding | REDs | finding | order |
|---|---|---|---|
| P-12 | `11705`, `11706`, `11707` | `11711` | RED first |
| STRIKE 3 | `11712`, `11713`, `11714` | `11717` | RED first |
| STRIKE 4 | same three | `11718` | RED first |
| STRIKE 5 | `11726` | `11729` | RED first |
| STRIKE 6 | `11727` | `11730` | RED first |
| STRIKE 7 | `11733`, `11734` | `11736` | RED first |

P-12 was struck UNBOUND at it.1 (C19) for having no logged event and no test.
Both are now present, the wording carries this office's correction of fact, and
both planted positives are GREEN in the same invocation style. **P-12 is BOUND.**

**Noted, not struck.** MARS's `done` at `11731` states "JUPITER it.2 did not land
inside the wall; not attacked", and STRIKE 7 is then filed at `11733`-`11736`,
after his own `done`. The report discloses the supersession in its own header and
at `:323-326`. The REDs still precede the finding; only the `done` is premature.

### Controls probed — none vacuous

Every GREEN control examined was made to fail by making true the thing it
controls for.

| control | probe | result |
|---|---|---|
| `test_control_the_isomorphism_is_exact_where_both_wings_are_defined` | one side scaled by `1 + 1e-6` | **FAILS**, max delta `8.999999999703689e-07` against a bar of `1e-12` |
| `test_control_saturns_draw_range_is_read_from_the_w3_cells` | threshold `100.0` to `120.0` | **FAILS**, `assert 116.00607299804688 > 120.0` |
| `test_control_the_gate_floor_predicts_the_zero_gate_count_on_every_cell` | `n_zero_gates > 0` to `>= 0` | **FAILS** on the escape cell `arm_smprime:t2:n2048:seed2` |

The isomorphism control is the one that concedes JUPITER's algebra. **It runs, it
passes, and its tolerance binds** — the perturbation clears the bar by nine
orders of magnitude. It is not comparing an expression to itself.
---

## THE LEDGER

Verdict `clean` means the claim is BOUND and its numbers reproduce. Verdict
`struck` means it is not — and says nothing about whether it is true. A struck
claim goes to Open and its author does not appeal.

| # | claim | author | verdict |
|---|---|---|---|
| C1 | `tests/jupiter/` 92 to 101 passed | Jupiter | clean |
| C2 | `test_v20_r15_it2_ldom_census.py` 9 nodes, 9 passed | Jupiter | clean |
| C3 | L-DOM RED is `ImportError: cannot import name 'domain_census_facts'` | Jupiter | clean |
| C4 | that RED took **8 nodes** / all 9 RED | Jupiter | **struck** |
| C5 | L-DOM census 0/384, 0/2048, 0/8 nilpotent | Jupiter | clean |
| C6 | `crude == exact`, two-sided, `phi = 1/27` | Jupiter | clean |
| C7 | planted negative: the two-sided **node** fires at `crude * 10.0` | Jupiter | **struck** |
| C8 | seeded permutation `default_rng(20152)` is named and reproduces | Jupiter | clean |
| C9 | N=1-primitive merge verdict | Jupiter | **struck** |
| C10 | six Lean withdrawals, written down and enumerated | Jupiter | clean |
| C11 | `test_v20_r15_wings_distinct.py` 5 passed | Saturn | clean |
| C12 | `tests/saturn` 59 passed / 7 failed, 6 pre-existing | Saturn | clean |
| C13 | distinctness verdict, W1 distinct from W2 and W3; N is 3 or 2 | Saturn | clean |
| C14 | that verdict was RED first **in the log** | Saturn | **struck** |
| C15 | K5 WITHDRAWN, written in the report | Saturn | clean |
| C16 | K6 halved 4 to 2, survivors `d=65` and `1.3e-3` | Saturn | clean |
| C17 | V-7b mechanism, written in the report | Saturn | clean |
| C18 | `tests/mars_v20/` 16 RED / 14 GREEN | Mars | clean |
| C19 | P-12 BOUND, 3 red / 3 green, both planted positives green | Mars | clean |
| C20 | STRIKE 3, the one fact, 3 red / 3 green | Mars | clean |
| C21 | STRIKE 4, `arm_pl` ships no cap | Mars | clean |
| C22 | STRIKE 5 bound, RED first | Mars | clean |
| C23 | STRIKE 5, `99.17%` of the gate is exactly `1.0` | Mars | clean |
| C24 | STRIKE 5, SATURN discloses it in his own `Limits` | Mars | clean |
| C25 | STRIKE 6 bound, RED first | Mars | clean |
| C26 | STRIKE 6, `25.0x` file order vs `8.4x` best-first | Mars | clean |
| C27 | STRIKE 7 bound, RED first | Mars | clean |
| C28 | STRIKE 7, `a_hat_max > 1.0` on 8/8, max `g = 4.7536` at seed 7 | Mars | clean |
| C29 | STRIKE 7, 5/3 partition at `0.662128` / `1.113339`, zero overlap | Mars | clean |
| C30 | control: isomorphism exact to `< 1e-12`, and it binds | Mars | clean |
| C31 | control: draw range read from the W3 cells, and it binds | Mars | clean |
| C32 | control: gate floor predicts the zero-gate count, and it binds | Mars | clean |
| C33 | Clopper-Pearson `5/8 -> 0.2449`, `1/8 -> 0.0032`, `0/8 -> 0.0` | coordinator | clean |
| C34 | sizing: only `8/8` clears `0.5`; `0.4735`, `0.3491`; first `N` is `65` | coordinator | clean |

**34 audited, 4 struck.**

Struck: **C4** (a collection error emits `1 error`, no per-node count, and the
file holds 9 not 8), **C7** (the planted negative re-declares the assertion on a
local and never invokes the node it names), **C9** (no test in `tests/jupiter/`
asserts the merge, so the suite-level RED cannot reach it), **C14** (SATURN filed
one composite event with no `t` field, so no RED precedes the verdict in the log).

Three of the four are JUPITER's, and all three are of one family: **a RED or a
mutation that is described more strongly than the tool actually reported.** C14 is
a different family and is bookkeeping, not evidence — his tests and his report are
sound; his log is one atom.

**Repaired since it.1, and recorded as such:** C7-of-it.1 (the unnamed
permutation) is now seeded and named; C19-of-it.1 (P-12 unbound) is now bound;
C20-of-it.1 (the one-sided `crude` node) is now two-sided; C25 and C26 of it.1
(the two K6 instances) are withdrawn by their author.

---

## TREE

Three sources mutated and reverted this check: `scripts/v20_m14_cheeger.py`
(renamed definition, then the seeded permutation injected into `repo_chain_weights`)
and the three `tests/mars_v20/` control files. Every one restored and verified —
`scripts/v20_m14_cheeger.py` md5 `cb162011c0c442ef3e0b80382793f452`, identical to
its pre-mutation capture; the three Mars files `diff`-empty against their backups.
`tests/mars_v20/` re-runs at 16 failed / 14 passed, and
`test_v20_r15_it2_ldom_census.py` at 9 passed, both identical to the baselines
taken at open.

**No git write of any kind. Nothing touched Kaggle.**

`git status --porcelain` at close differs from its state at open by five paths.
This office accounts for all five and claims only one:

- `V20_R15_IT2_INSPECTOR.md` — **this file, written by this office.**
- `V20_R15_IT3_MERCURY.md`, `tests/mercury/arena_price.py`,
  `tests/mercury/test_v20_r15_it3_arena_price.py`,
  `tests/saturn/test_v20_r15_wings_distinct_percell.py` — **not written by this
  office and not touched by any nurse it sent.** They carry it.3 / MERCURY names
  and appeared mid-check; a concurrent it.3 process is writing into this working
  tree while this audit ran. Reported rather than absorbed into a "tree
  unchanged" line that would not be true.

Everything this office mutated is byte-for-byte as it was found. The `M`
entries — `house-events.jsonl` (appended to, as the protocol requires) and
`pytest.ini` — are unchanged from open apart from this check's own `audit`
events.
