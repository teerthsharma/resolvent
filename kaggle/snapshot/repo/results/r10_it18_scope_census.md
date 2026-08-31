# The it.18 scope census: 31 controls, 19 fire, and 2 of the firings are defects

MARS attack #2, v-main.3M script iteration 18. The script line, verbatim:

> MARS: attack #2 scope-hunt — for each control, the reachability set
> R(control) vs production domain D; fires iff R ∩ D = ∅ for any control (the
> 14th class, mechanized as a set census).

Code `scale/r10_it18_scope_census.py`, rows `results/r10_it18_scope_census.jsonl`.
Reproduce every number below with

```
python -m scale.r10_it18_scope_census            # demo(), assert-based, ~35 s
python -m scale.r10_it18_scope_census --write    # the census, 53.5 s, writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, no torch, Windows 11 x86-64, branch
`feat/r9-causal-consequence` at `bf2a769`. One process, no training, no capacity
sweep, no GPU work. `D_shipped` is built through the front door on the corpus's
own seeds, `0x3a140000 + i` and `0x3a150000 + i`, printed per instance in the
it.14 and it.15 rows this census reproduces.

`scale.vram_gate.preflight(0, name='r10-it18-scope-census', host_mib=512)`
returned `HOST FITS -- needs 640 MiB, 2141 MiB available` for the `--write` run
and `2640 MiB available` for the `demo()` before it. Both are printed; the box
was at `13,950 MiB in use` of `16,091` when the census ran and neither reading is
the friendlier one from earlier in the session.

---

## 0. THE HEADLINE, BEFORE THE EVIDENCE

| | |
|---|---|
| **controls examined** | **31** |
| **controls firing** (`R ∩ D = ∅`) | **19** |
| firing, **proven** empty | **18** |
| firing, **sampled** only | **1** |
| **firing AND a defect** (certifying role) | **2** |
| errored | **0** |

1. **The firing rule as written flags 19 of 31 controls, and 17 of those 19 are
   not defects.** A refusal guard is exercised on inputs production cannot emit
   *because that is what a refusal guard is for*. The rule needs the control's
   ROLE, and the role cannot be inferred by the census — it is declared. §2.
2. **The two real hits are `it14/path_case` and `it15/probe-ring-B2`**, both
   **certifying** controls whose emptiness is **proven**, not sampled. Two of
   it.14's four oracle-agreement rows and it.15's "the route is not broken"
   control rest on graphs the production builder cannot emit. §4.
3. **The proven/sampled distinction is not decoration and it is exercised.**
   18 firings are proven by a violated necessary invariant or by exhaustive
   membership in a finite domain. **One is sampled only** — a relabelled real
   instance that violates nothing and was not found in 400 production draws.
   For that one the census **cannot** distinguish `R ∩ D = ∅` from
   `R ∩ D` small and unsampled, and says so. §5.
4. **The round law "bind a must-fire to a constructed input" and this clause
   point in opposite directions.** A must-fire bound to a constructed input is,
   by construction, a control whose `R` may miss `D`. The role taxonomy is what
   reconciles them; without it the two rules are in direct conflict. §8.

---

## 1. HOW `R` AND `D` WERE MADE FINITE

A literal intersection of two infinite input spaces is not computable, so both
sides are replaced by named finite objects. **The replacement is the thing to
attack, not the arithmetic**, so it is stated first and in full.

### 1.1 `R(control)` — exact, no proxy

`R(control)` is replaced by **the inputs the control is actually handed**. Every
control in the registry is deterministic: its inputs are either literals in the
source or are generated from a declared seed. `R` is therefore a finite,
enumerable set and **no proxy is involved on this side**. `R_size` runs 1 to 28
and is printed per row.

This is a narrowing, and naming it is the point: `R` as "the set of inputs the
control *could ever* be handed given where it enters" is larger than "the inputs
it *is* handed". The census measures the smaller set. A control whose `R` is
larger than what it is handed — one parameterised by a seed a caller could
change — would need its parameter range declared before this census means what
it says. **No control in this registry has such a parameter**; all 31 are called
at fixed arguments.

### 1.2 `D` — two objects, and every verdict says which one decided it

**`D_shipped`** — the instances the production builder has actually emitted:
`dual.reproduce_arm()` + `dual.widen_arm()`, entered through the **front door**,
**28 instances** plus **1 band miss that carries no instance**. Membership is
`adm.labelled_key`, SHA-256 over the sorted edge list on the generator's own node
labels plus the boundary list plus the float64 bytes of `g` — imported from
it.16, not re-implemented. **Exact in both directions**: a match proves
membership, and since this set is finite, a non-match proves non-membership *of
this set*.

**`D_invariant`** — five **necessary** conditions on membership of `D`, each read
off the production code and cited to the line that forces it:

| | condition | forced by |
|---|---|---|
| **I1** | `n` is one of the shapes production is called at, `{64, 128, 256, 512, 1024}` | `scale/r10_corpus_spec.py:506` `RUNGS`; `scale/r10_dual_oracle.py:124-125` `WIDEN_SHAPES` / `WIDEN_BIG` |
| **I2** | edge count is `(n-1) + extra_for(n, d)` for some shipped `d` at that `n` | `scale/r10_admissibility.py:168`; `scale/r10_corpus_spec.py:138` |
| **I3** | the graph is connected | `scale/r10_corpus_spec.py:145-152`, attachment tree, connectivity by construction |
| **I4** | `g ∈ [0, 1)^{\|B\|}` | `scale/r10_dual_oracle.py:325` `rng.uniform(0.0, 1.0)` |
| **I5** | `lambda_2(adj, B) ∈ [0.90, 0.95]` | `scale/r10_corpus_spec.py:361` `stratify` returns `k=None` outside the band; `scale/r10_dual_oracle.py:317-323` emits `band_miss` |

**They are necessary and never sufficient, and the census refuses to treat them
otherwise.** An input violating one is **proven** out of `D`. An input violating
none is **not** thereby in `D` — it is UNDECIDED, and falls to §1.3. Promoting
"violates nothing" to "is a member" is exactly the error a scope census exists to
catch, so the code does not do it.

### 1.3 The fallback, and it is the weak one

An undecided input gets a **sample**: `SEARCH_DRAWS` production graph draws at the
control's own `n`, compared by `adm.graph_key`. Graph identity is settled by
`spec.draw_graph` alone — `stratify` draws nothing — so the sample needs no
eigensolve and searches the object that decides membership first.

**`SEARCH_DRAWS = 200`, imported not chosen**: it is the draw count
`scale/r10_corpus_spec.py:742` already uses to price a property against the
production draw (`naive_yield(200, SEED0 + 0x1000)`), lifted out of that file's
AST so a drifted copy is impossible. Seeds are `0x3a180000 + i`, declared in the
module header before the census ran. Narrowing the search to the control's own
`n` makes it **cheaper and therefore weaker**, and that is said here rather than
in a footnote: a miss is evidence about that `n` only.

### 1.4 Three further domains, because three controls are not instances

Not every control's input is a corpus instance, and lumping them in would have
been the easy dishonesty. Each gets its own `D` and its own decision procedure.

- **`D_label`** — labels production can emit on a given instance. `u` is the
  harmonic extension, so `y` is producible iff `(I - P_II) y` lies in the column
  space of `P_IB` at some `g ∈ [0,1)`. Least squares gives the only candidate
  `g`; a residual above the imported `AGREEMENT_TOL = 1e-10` is a **proof** that
  no `g` produces `y`. **Proven in both directions.**
- **`D_resource`** — the preflight requests production issues. Taken from the AST
  of every `scale/*.py`, **36 preflight arguments across the package**, each
  classified literal or not. `D_resource` is the literal ones; a request computed
  from a live reading is provably not among them. *(A line-oriented scan would
  have found zero here: both it.16 resource plants are wrapped across lines, and
  an empty domain reported as a clean one is the exact failure this census is
  about.)*
- **`D_file`** — files under the repository this campaign writes. Decided by
  path prefix. **Proven.**

---

## 2. THE FIRING RULE AND THE ROLE TAXONOMY, BOTH FILED BEFORE MEASURING

```
FIRES(control)  iff  no element of R(control) is IN D.
strength = "proven"   iff every element is out by a violated invariant
                          or by exhaustive membership in a finite domain
         = "sampled"  iff any element's out-ness rests only on the draw search
```

**And the taxonomy, filed in the same commit, before any control was scored**,
because the rule alone gives the wrong answer 17 times out of 19:

| role | what it means | what firing means |
|---|---|---|
| **certifying** | its reading supports a claim ABOUT PRODUCTION | **a defect.** The claim rests on inputs production cannot make. |
| **excluding** | it exercises a REFUSAL | **expected.** Production cannot emit what a guard exists to refuse. The dual obligation is a non-firing in-domain control. |
| **illustrating** | it exists to demonstrate a hazard | **the result.** Firing is what it was built to do. |

**Role is declared per control in the registry, beside the site that owns it, and
the census cannot infer it.** That is a limitation, not a design choice: whether
a reading is used to certify production is a fact about the report that quotes
it, not about the input. A registry with a role mislabelled produces a wrong
verdict and this census would not notice.

`status = "failed"` iff `fires AND role == certifying`. `demo()` asserts the rule
both ways on the same firing control: scored `ok` as `excluding`, `failed` as
`certifying`.

---

## 3. THE CENSUS

31 controls, from the four round-10 corpus modules, each with the site that owns
it. `R` and `in_D` are counts of inputs.

| control | site | role | domain | `\|R\|` | in `D` | fires | strength |
|---|---|---|---|---|---|---|---|
| `it14/must_fire_oracle` | `r10_corpus_spec.py:432` | excluding | instance | 1 | 1 | no | — |
| `it14/must_fire_dose` | `:441` | excluding | instance | 1 | 1 | no | — |
| `it14/admissible-disconnected` | `:711` | excluding | instance | 1 | 0 | **yes** | proven |
| `it14/admissible-isolated` | `:717` | excluding | instance | 1 | 0 | **yes** | proven |
| **`it14/path_case`** | **`:478`** | **certifying** | instance | 1 | **0** | **yes** | **proven** |
| `it15/probe-B1` | `r10_dual_oracle.py:390` | excluding | instance | 1 | 0 | **yes** | proven |
| `it15/probe-split` | `:390` | excluding | instance | 1 | 0 | **yes** | proven |
| **`it15/probe-ring-B2`** | **`:390`** | **certifying** | instance | 1 | **0** | **yes** | **proven** |
| `it15/plant-absorbing` | `:217` | excluding | instance | 1 | 1 | no | — |
| `it15/plant-kirchhoff` | `:217` | excluding | instance | 1 | 1 | no | — |
| `it15/plant-shared` | `:217` | excluding | instance | 1 | 1 | no | — |
| `it15/demonstrate_halt` | `:419` | excluding | instance | 3 | 3 | no | — |
| `it16/plant-leak` | `r10_admissibility.py:661` | excluding | instance | 1 | 1 | no | — |
| `it16/plant-ring32` | `:665` | excluding | instance | 1 | 0 | **yes** | proven |
| `it16/control-clean-SPLIT-ARM` | `:670` | certifying | instance | **28** | **28** | no | — |
| `it16/probe-F0..F3` | `:675` | certifying | instance | 1 | 1 | no | — |
| `it16/plant-linear-label` | `:681` | excluding | label | 1 | 0 | **yes** | proven |
| `it16/control-noise-label` | `:690` | excluding | label | 1 | 0 | **yes** | proven |
| `it16/plant-host` | `:711` | excluding | resource | 1 | 0 | **yes** | proven |
| `it16/plant-vram` | `:715` | excluding | resource | 1 | 0 | **yes** | proven |
| `it17/control_front_door` | `r10_it17_battery.py:191` | certifying | instance | 1 | 1 | no | — |
| `it17/control_below_door` | `:203` | certifying | instance | 1 | 1 | no | — |
| `it17/control_hand_built-star` | `:215` | illustrating | instance | 1 | 0 | **yes** | proven |
| `it17/plant-dead-bit` | `plant_dead_boundary_bit` | excluding | instance | 1 | 0 | **yes** | proven |
| `it17/plant-shuffled-label` | `demo step 3b` | excluding | label | 1 | 0 | **yes** | proven |
| `it17/planted-key-file` | `demo step 5` | excluding | file | 1 | 0 | **yes** | proven |
| `it17/clean-control-file` | `demo step 5` | excluding | file | 1 | 0 | **yes** | proven |
| `it17/C-E-ii-live-scan` | `scan_population` | certifying | file | 1 | 1 | no | — |
| `it18/plant-two-components` | this module | illustrating | instance | 1 | 0 | **yes** | proven |
| `it18/plant-g-out-of-range` | this module | illustrating | instance | 1 | 0 | **yes** | proven |
| **`it18/plant-relabelled`** | this module | illustrating | instance | 1 | 0 | **yes** | **sampled** |

| by role | examined | firing |
|---|---|---|
| certifying | 7 | **2** |
| excluding | 20 | 13 |
| illustrating | 4 | 4 |

| by domain | examined |
|---|---|
| instance | 23 |
| label | 3 |
| resource | 2 |
| file | 3 |

### 3.1 The split inside `excluding` is the census's own sanity check

**7 of the 20 excluding controls do not fire, and the reason is mechanical**: a
plant on the **instrument** — `must_fire_oracle` corrupting the oracle,
`must_fire_dose` shifting `lambda_2`, `check_instance`'s three plants,
`demonstrate_halt`, `plant-leak` duplicating a row — leaves the **input** in
domain. A plant on the **input** takes it out. The census separates those two
without being told which is which, and a census that returned "everything fires"
would have been worth nothing.

---

## 4. THE TWO DEFECTS

### 4.1 `it14/path_case` — three of it.14's agreement rows are out of domain

`scale/r10_corpus_spec.py:478`. The path `P_{L+2}` with the two endpoints as `B`
and `g = (1, 0)`. **Proven out of `D`: violates I1 (shape), I2 (edge count) and
I4 (`g` range).** `n = 9` is not one of the five shapes production is called at;
`8` edges is not `(n-1) + extra_for(n, d)` for any shipped `d` at `n = 9`,
because there is no shipped `d` at `n = 9` at all.

The reading it certifies is it.14 §2's agreement table. **Three of its six rows
rest on this control alone**: `A vs C (Kirchhoff), 4 path cases 2.220446e-16`,
`A vs closed form u_x = 1 - x/(L+1) 2.220446e-16`, and
`lambda_2 vs cos(pi/(L+1)), 4 path cases 2.220446e-16`. All three are
certifications on graphs the production builder cannot emit.

**They are redundant rather than absent, and that is the mitigating fact.** The
same oracle pair is checked in domain: `A vs B, 12 rungs, 9.992007e-16` (it.14
§2) and `A vs Kirchhoff, 29 instances, 1.554312e-15` (it.15 §2). So the
certification exists in domain; what does not exist in domain is the *closed
form*, and the closed form is what the path case is for. There is no closed-form
harmonic extension for a stratified attachment-tree-plus-chords graph, so the
control cannot be moved into domain — it can only be labelled.

### 4.2 `it15/probe-ring-B2` — the "route is not broken" control

`scale/r10_dual_oracle.py:390`. The 6-node ring, `|B| = 2`, `g = (1, 0)`. **Proven
out of `D`: violates I1, I2, I4 and I5.** Its own comment names its job: *"the
control: same ring, `|B| = 2`, applicable and expected to pass, so the two
refusals above are not a broken route."*

**Its two siblings, `probe-B1` and `probe-split`, are correctly out of domain and
the author said so** — `probes()`'s docstring reads, before this census existed,
that *"the drawn corpus never hits it — `draw_graph` is connected by construction
and the stratifier never returned `|B| < 4`"*. That is a declaration of
`R ∩ D = ∅` in prose, and the census confirms it mechanically for both. The
finding is the **third** probe: the control that certifies the route is healthy
is run only on the same out-of-domain ring, so a route that worked on 6-rings and
failed on the corpus would pass it.

**Again redundant rather than absent**: it.15's headline is the same route
agreeing with the absorbing oracle on 29 in-domain instances at `1.554312e-15`.
The control is not load-bearing. It is, as written, unable to bear load.

### 4.3 A note on I4, so the two verdicts are not overstated

Both defects trip **I4** (`g ∈ [0,1)`) because both hand-set `g = (1, 0)` and
`rng.uniform(0.0, 1.0)` excludes `1.0`. That invariant is *true* and it is
*thin*: it would flag any hand-written `g` regardless of the graph. **The
substantive violations for both controls are I1 and I2**, which are about the
graph and cannot be repaired by rescaling a boundary value. The verdicts stand on
I1 and I2; I4 is reported because it fired, not because it carries the argument.

---

## 5. PROVEN AGAINST SAMPLED, WHICH IS THE WHOLE CLAUSE

**18 of the 19 firings are proven, by four different decision procedures:**

| how | firings |
|---|---|
| a violated necessary invariant (`D_invariant`, instance domain) | **11** |
| a least-squares residual certifying that **no** `g` produces the label (`D_label`) | **3** |
| the request is not a literal production issues (`D_resource`) | **2** |
| the path is outside the repository root (`D_file`) | **2** |
| **proven, total** | **18** |
| the draw search missed (`D` sampled) | **1** |

The label domain's three are the strongest of the four: the residual is a
certificate that no `g` whatsoever produces that label, not merely that none was
found.

**One firing is sampled: `it18/plant-relabelled`.** It is a real front-door
instance with two node labels transposed and the boundary carried along, so it
has the same `n`, the same edge count, the same connectivity, the same `g` and
the same `lambda_2` — **it violates no invariant** — and its graph key was not
found in **400** production draws at `n = 64` (200 seeds × the two shipped mean
degrees at that `n`).

> **For this control the census cannot distinguish `R ∩ D = ∅` from `R ∩ D` being
> small and unsampled, and it does not claim to.** A relabelled draw is in the
> support of `spec.draw_graph` — the generator draws a uniform permutation, so
> the transposed graph has exactly the same probability as the original. The
> honest statement is "not observed in 400 draws", and the row says `"strength":
> "sampled"` rather than `"proven"`.

That control exists **for this purpose**: without it the sampled branch of the
census would never have executed, and a branch that never runs is the failure
mode this round has already catalogued once. It is constructed, so its
reachability does not depend on anything in the live tree.

**Both defects in §4 are proven, not sampled.** That matters: the two verdicts
this census actually turns on are the two that rest on the strong half of the
mechanization.

---

## 6. THE CENSUS SHOWN ABLE TO FIRE, AND NOT TO

All five are asserted in `demo()` and all are bound to constructed inputs.

| | input | result |
|---|---|---|
| **must NOT fire** | a front-door instance | in `D`, **proven** by labelled-key match |
| **must fire, I3** | 64 nodes, the production edge count for mean degree 4, two components | out, proven — `I3 connected`, **and `I5`**: splitting the graph moves `lambda_2` out of the band too, so it is a two-invariant plant and is reported as one |
| **must fire, I4** | a real instance with `g ≡ 7.5` | out, proven, **`I4` alone** — the single-invariant plant on this census |
| **must fire, I1** | a 6-node ring | out, proven — `I1 shape`, `I2 edge count`, `I5` |
| **sampled branch** | the relabelled instance | out, **sampled**, 400 draws |
| **label domain, both ways** | the real `u`; the shuffled `u` | in; out, proven, residual `1.436e-01` against `1e-10` |
| **role rule, both ways** | the same firing resource control | `ok` as excluding, `failed` as certifying |

---

## 7. COUNTS, SIX COLUMNS

Over the 33 rows written before the summary row itself (1 provenance, 1 band
miss, 31 census):

| | count |
|---|---|
| **passed** | **29** |
| **failed** | **2** (`it14/path_case`, `it15/probe-ring-B2`) |
| **errored** | **0** |
| **inapplicable** | **0** |
| **not reached** | **1** |
| **unscored** | **1** (the provenance row, which carries no status and is not folded into a pass) |

`not_reached` is `widen-n128-d4-t0.905`, seed `0x3a150000` — it.15's band miss,
an *attempt* that produced no instance. It is carried so that `D_shipped` is
reported as **28 instances from 29 attempts** rather than as 29.

---

## 8. THE AMENDMENT

> **Amendment (it.18).** *The firing rule `R ∩ D = ∅` is necessary but not
> sufficient for a defect. It must be conditioned on the control's declared
> ROLE: firing is a defect only for a control whose reading certifies a claim
> about production. A control that exercises a refusal must be out of domain, and
> its obligation is not to be in domain but to be paired with an in-domain
> control that does NOT fire.*

**Applied to this registry the amendment moves 17 verdicts.** Without it the
census reports 19 defects, of which 13 are refusal guards doing their job and 4
are hazard demonstrations whose entire content is that they are out of domain.

**This puts the clause in direct tension with a standing round law.** The law
that a must-fire be *bound to a constructed input*, so its reachability cannot
quietly depend on live data, forces exactly the shape this clause fires on. Both
rules are right; they apply to different roles, and neither says so on its own.
That is what the taxonomy is for, and it is why the taxonomy is declared rather
than inferred.

A second, smaller amendment falls out of §1.1: **`R` should be the set of inputs
a control CAN be handed, not the set it IS handed.** They coincide for all 31
controls here because all 31 are called at fixed arguments, but a control taking
a caller-supplied seed range would need that range declared before this census
means what it says.

---

## 9. LIMITS

Collected here rather than scattered.

**Role is declared, not measured.** A control whose role is mislabelled in the
registry gets the wrong verdict and this census will not notice. The registry was
built by reading the four modules' `demo()` and must-fire bodies; the site is
printed with every row so the list can be audited against the source rather than
trusted, but the list itself is hand-enumerated and a control nobody noticed is a
control this census did not examine. **31 is the count of controls found, not a
proof that there are 31.**

**The five invariants are necessary and not sufficient**, so "in `D`" is only ever
established by exact membership in `D_shipped` — a finite set of 28. Any
production instance not in that 28 would be UNDECIDED under this census, which is
why `it16/control-clean-SPLIT-ARM` reads `28 / 28` and not more.

**The sample is 200 seeds at one `n`**, narrowed for cost, and is evidence about
that `n` only. It was spent on exactly one control. A registry with more
undecided controls would cost proportionally more and would still only ever
produce "not observed".

**`D_resource` is decided by whether an argument is a literal.** That is
mechanical and it is a proxy for "production issues it": a production call site
that computed a request from a reading would be misclassified, and there is no
such call site today across the 36 arguments scanned. **`D_label` assumes the
label is the harmonic extension**, which is the only label this corpus declares;
a corpus that declared a second label functional would need a second predicate.

**`D_shipped` is built through the front door and costs 52.2 s** for both arms,
which is why it is built once and shared across all 31 controls rather than per
control. Nothing here re-measures it.

No claim is made about controls outside round 10's four corpus modules, and none
about what a model can learn from this corpus. This iteration adjudicates scope
and nothing else.
