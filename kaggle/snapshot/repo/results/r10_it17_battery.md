# The it.17 battery: C-D holds, C-E holds twice, C-F fails on every rung and on one bit by exactly zero

MARS, v-main.3M script iteration 17. The script line, verbatim:

> SATURN: C-D = exists arm with NRMSE < 1.0 before any failure is scored; C-E
> oracle executable, zero answer keys in files; C-F do()-bit flip moves labels
> with effect >= pre-set delta, control inputs drawn from the PRODUCTION batch
> path (L-SCOPE).

Code `scale/r10_it17_battery.py`, rows `results/r10_it17_battery.jsonl`.
Reproduce every number below with

```
python -m scale.r10_it17_battery            # demo(), assert-based, ~30 s
python -m scale.r10_it17_battery --write    # the battery, 130.6 s, writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, no torch, Windows 11 x86-64, branch
`feat/r9-causal-consequence` at `bf2a769`. One process, no training, no capacity
sweep, no GPU work. The corpus is entered through the production builder, so the
seeds are JUPITER's `0x3a140000 + index` and every row prints its own. The only
seeds this battery adds are `PROBE_SEED = 0x3a160000` (it.16's, imported) and
`CD_PLANT_SEED = 0x3a170001`, declared in the module header before the plant was
run.

`scale.vram_gate.preflight(0, name='r10-it17-battery', host_mib=512)` returned
`HOST FITS -- needs 640 MiB, 1960 MiB available` at the start of the `--write`
run and `7344 MiB available` at the start of the first `demo()` — the box moved
by 5.4 GiB between them with a sibling job running, and both readings are printed
rather than the friendlier one. Peak is one `960 x 960` `eigh` inside `stratify`
plus one `944 x 944` LU with 80 right-hand sides.

---

## 0. THE HEADLINE, BEFORE THE EVIDENCE

| clause | verdict | the number |
|---|---|---|
| **C-D** an arm reads NRMSE < 1.0 | **HOLDS, 12 of 12** | `khop_128` on `reproduce-n128-d4-t0.95`, **`0.005092`**, against a mean-predictor control at exactly `1.000000` |
| **C-E(i)** the oracle is executable | **HOLDS, 12 of 12** | sensitivity to `g` `2.29e-01`…`6.93e-01`, determinism `0.0`, second route `1.554e-15` |
| **C-E(ii)** zero answer keys in files | **HOLDS as written** — and the scan is not empty | **0** stored label vectors in **5,461 files / 142,271,756 bytes examined**; **56** two-coordinate disclosures in one file |
| **C-F(a)** the do()-bit moves labels by `delta` | **FAILS, 12 of 12**, and one bit moves by **exactly `0.0`** | filed `delta = 0.5`; measured `fd_min` `0.000000`…`0.359833`; **1 of 324 do()-bits clears it** |
| **C-F(b)** controls enter at the front door | **the instrument works; the round's own path does not use it** | classifier agrees with `p1prime` on **191/191** files; the 12 controls this round has been using enter **below** `make_rung`, leaving 5 stages unexercised |

1. **C-D is established, and the arm that establishes it is not the one that
   comes cheapest.** `khop_128` — 128 rounds of the message passing the
   architecture is supposed to do, untrained — reads `0.005092`. The it.16
   linear probe also reads under `1.0` on 10 of 12 rungs, and **that is the same
   reading that fails C-B**. §1.3.
2. **The k-hop arm does not cross `1.0` until between `k = 32` and `k = 64`, on
   every rung.** At `k = 32` its *best* cell in the whole corpus still reads
   `1.098575` — worse than predicting the mean. The corpus's label has a large
   mean and a small spread, so a truncated arm pays for the mean before it earns
   any of the variance. §1.2.
3. **`reproduce-n1024-d4-t0.95` (seed `0x3a14000a`) ships a boundary node whose
   do()-bit moves the label by exactly `0.0` in every norm.** Not a plant. It is
   a boundary node all of whose neighbours are also boundary nodes, and nothing
   in `spec.stratify` forbids one. §3.2.
4. **A hand-built control turns C-F from `failed` into `ok`.** The star graph
   reads `fd = 2.000000` where every front-door instance reads under `0.36`. No
   production builder can emit that star. That is the L-SCOPE clause earning its
   keep, in this corpus's own units. §4.3.

Counts, six columns, in §5.

---

## 1. C-D — AN ARM READS NRMSE < 1.0

### 1.1 The bar is not a choice, and the control proves it

`NRMSE < 1.0` is not a threshold anybody picked: `rips_gate.nrmse`
(`scale/rips_gate.py:118-127`) normalises by `sd(y)`, so the mean predictor reads
**exactly** `1.0`. The `mean` arm is carried in every row for that reason and
reads `1.000000` on all 13 C-D rows to `< 1e-12`; a row where it did not would be
measuring something that is not NRMSE, and `demo()` asserts it before anything
else is read.

### 1.2 The arms, declared before any of them was run

Three families, all reading only `(adj, B, g)` — the input a consumer is handed
— and none of them touching `u`:

| arm | what it is |
|---|---|
| `mean` | predict the mean. The control. Exactly `1.0` by construction. |
| `probe_F3` | it.16's linear probe on its `F3_boundary_data` ladder rung, `adm.probe`, held-out half/half at `PROBE_SEED`, imported not re-implemented. |
| `khop_k` | `u_k = sum_{m<k} P_II^m P_IB g`, `k` in `rips_gate.LADDER_KS`. Untrained, deterministic, exactly `k` rounds of message passing. |

**`KHOP_KS` is `rips_gate.LADDER_KS` entire.** An earlier cut at `k <= 32` was
written and then removed, because the crossing turned out to sit on the far side
of it — a truncated ladder that stops just before the answer is a ladder chosen
to flatter the verdict, and the removal is recorded here rather than in a commit
nobody reads.

`reproduce-n64-d4-t0.95`, the whole ladder, one row of the `.jsonl`:

| `k` | 1 | 2 | 4 | 8 | 16 | 32 | **64** | **128** |
|---|---|---|---|---|---|---|---|---|
| NRMSE | `8.9847` | `8.4548` | `7.5275` | `5.9707` | `3.7511` | `1.4782` | **`0.2295`** | **`0.0055`** |

**Monotone, and it crosses `1.0` between `k = 32` and `k = 64`.** That is worth
stating plainly: *sixteen rounds of message passing on this corpus is worse than
predicting the mean, and so is thirty-two.* The reason is not the dose theorem
failing — it.14 §4.2 measured `err(t) <= lambda_2^t` holding tightly in the
degree-weighted 2-norm — it is that NRMSE divides by `sd(u)` and this corpus's
`u` sits in a narrow band about a large mean (`[0.2947, 0.5700]` on this rung).
The truncation error is small against `||u||` and large against `sd(u)`.

### 1.3 The reading, and the arm that establishes it

| arm | min over 12 rungs | median | max | cells under `1.0` |
|---|---|---|---|---|
| `mean` (control) | `1.000000` | `1.000000` | `1.000000` | **0 / 12** |
| `probe_F3` | `0.566979` | `0.745818` | `1.416857` | **10 / 12** |
| `khop_16` | `2.691092` | `4.739319` | `7.530813` | 0 / 12 |
| `khop_32` | `1.098575` | `2.013902` | `3.262199` | **0 / 12** |
| `khop_64` | `0.183164` | `0.382748` | `0.612136` | **12 / 12** |
| **`khop_128`** | **`0.005092`** | `0.013580` | `0.021554` | **12 / 12** |

**The crossing is between `k = 32` and `k = 64` on all twelve rungs, not on
average**: `khop_32`'s corpus-wide *minimum* is `1.098575` and `khop_64`'s
corpus-wide *maximum* is `0.612136`, so no rung crosses earlier and none crosses
later.

> **C-D: HOLDS. Arm `khop_128`, cell `reproduce-n128-d4-t0.95`
> (`n = 128`, mean degree 4, seed `0x3a140002`, `lambda_2 = 0.9455579636`,
> `|B| = 11`, `|I| = 117`), reading `NRMSE = 0.005092` against a mean-predictor
> control of `1.000000`.**
> `results/r10_it17_battery.jsonl`, `kind = "C-D"`, `id = "reproduce-n128-d4-t0.95"`.

**So the failure tables of it.16 are admissible in retrospect, and they were
published before this was checked.** That is the whole reason the clause exists,
and the order in which it was actually satisfied is recorded rather than tidied.

### 1.4 The cheapest satisfier is the one C-B forbids

`probe_F3` reads under `1.0` on 10 of 12 cells. It would satisfy C-D on its own.
It is also, cell for cell, **exactly the reading it.16 §3.3 reports as C-B's
failure** — the linear probe at `0.7639` median against a `FAIL_BAR` of `0.9`
that says the probe must be doing essentially nothing.

The two clauses are not contradictory — C-B is about a *trivial* decoder and C-D
about *some* decoder — but they are satisfied by the same number, and a corpus
whose only C-D witness is its C-B violation has not shown that anything but the
trivial probe can succeed. **That is why C-D is taken on `khop_128` here.** The
`khop` arm is the honest witness: it is not local, it is not linear, it needs
more than 32 hops, and it is exactly the computation the architecture under test
is supposed to perform.

### 1.5 C-D shown able to fail, on a constructed label

The plant is the same instance with its label vector permuted by
`CD_PLANT_SEED = 0x3a170001`, declared in the module header before the plant was
run. The input is untouched, so every arm is handed exactly what it was handed
before; only the correspondence is destroyed.

| | clean | shuffled |
|---|---|---|
| `reproduce-n64-d4-t0.95`, best arm | `khop_128` `0.005531` | `khop_128` **`1.421956`** |
| `reproduce-n128-d4-t0.95`, best arm | `khop_128` `0.005092` | `probe_F3` **`1.053683`** |
| verdict | **ok** | **failed** |

`1.421956` is `sqrt(2)` to three figures, which is what a random permutation of a
label predicts exactly: `E||y - pi y||^2 = 2 n var(y)`. The plant is bound to a
constructed input and its reachability does not depend on anything in the live
tree. `demo()` asserts both the `khop` reading crossing back above `1.0` and the
clause's verdict flipping to `failed`.

---

## 2. C-E — TWO SEPARATE PROPERTIES, MEASURED SEPARATELY

### 2.1 (i) The oracle is a FUNCTION, and three properties say so

The oracle is `spec.resolvent_oracle` (`scale/r10_corpus_spec.py:254`), one line:

```python
return np.linalg.solve(np.eye(PII.shape[0], dtype=np.float64) - PII, PIB @ g)
```

It takes `(P_II, P_IB, g)` and nothing else. There is no path by which a stored
label reaches it. Two further routes exist and are also functions of the input
alone: `spec.averaging_oracle` (`:259`), which iterates the defining mean-value
map straight off the adjacency lists and forms no matrix; and
`dual.kirchhoff_extension` (`scale/r10_dual_oracle.py:170`), the matrix-tree
cofactor route.

**Citing a signature is not a measurement, so three properties are measured per
rung:**

| property | what it rules out | reading, 12 rungs |
|---|---|---|
| **sensitivity to `g`** | a cached vector, which returns the same answer for every input | `2.286245e-01` … `6.934776e-01`, all `>>` `1e-10` |
| **determinism** | a stochastic or clock-reading oracle | **`0.0` on 12 of 12**, bitwise |
| **second route** | a wrong function that is nonetheless a function | `2.776e-16` … `1.554e-15`, five orders inside the tolerance |

The tolerance is `AGREEMENT_TOL = 1e-10`, `kirchhoff.AGREEMENT_TOL`, imported by
object identity as in it.14 and it.15 and not re-derived here. The second route
ran `7`…`80` cofactor solves per rung, 324 in total, and its own internal check
— the absorption probabilities summing to one — reads `4.44e-16` … `2.22e-15`.

**Executability is also shown end to end**: every one of the 28 shipped it.15
rungs is regenerated from four scalars `(n, mean_deg, seed, |B|)` and its label
recomputed, with `recovery_max_range_delta = 0.0` against the shipped
`u_absorbing_range`. Zero, not "small".

**C-E(i) shown able to fail.** The plant is an oracle that ignores its arguments
and returns the label vector it was constructed with. Everything downstream of it
is numerically perfect — it *is* the right answer — and only the sensitivity
property refuses it: `sensitivity_to_g = 0.0 <= 1e-10`, **fires**. `demo()`
asserts the refusal.

### 2.2 (ii) The answer-key scan, with its denominator

A search that found nothing is worth nothing without the count of what it looked
at, so the population is stated first.

| | files |
|---|---|
| walked | **46,557** |
| excluded by declaration: `lean/.lake` (vendored Lake build cache — mathlib source and `.olean` artifacts, nothing this campaign writes) | **35,950** |
| excluded by declaration: `.git` | **3,055** |
| skipped, not UTF-8 decodable | **2,089** |
| skipped, over 16 MiB | **2** |
| **examined** | **5,461** |
| **bytes read** | **142,271,756** |

Two independent routes over that same population, against the label vectors of
all **28** shipped it.15 rungs (`|I|` from 57 to 960), with **280** search
literals:

- **STRUCTURAL.** Every list of `>= 16` numbers in every `.json` / `.jsonl`,
  compared coordinate-wise against a label vector of matching length inside
  `AGREEMENT_TOL`. This is what a stored answer key *is*.
  **Result: 0 hits.**
- **TEXTUAL.** Any literal rendering — `repr` and `%.12g` — of coordinate 0, the
  middle coordinate, the last, the **min** and the **max** of any label vector,
  anywhere in any decodable file. **Result: 56 hits, every one of them in
  `results/r10_it15_dual_oracle.jsonl`.**

**The 56 are exactly `28 x 2`: the `u_absorbing_range` field, which ships
`[u.min(), u.max()]` at full float64 precision for every rung.** Coordinates 0,
middle and last were searched with the same needles and appear **nowhere**, which
is what makes the 56 a measurement of a known field rather than an accident.

> **C-E(ii): HOLDS on the clause as written — zero answer keys, where an answer
> key is a stored label vector. And it holds with a measured surface, not a
> zero: 2 of `|I|` label coordinates per rung are on disk, `3.5 %` of the label
> at `n = 64` and `0.21 %` at `n = 1024`.**

That surface is what makes the recovery binding of §2.1 possible in the first
place — it.16 uses it as its fingerprint — so removing it costs something.
Recorded, not repaired: if the property wanted is literally zero disclosure, the
smallest amendment is to ship a hash of the label vector instead of its extremes,
which fingerprints just as well and discloses nothing.

**C-E(ii) shown able to fail, bound to a constructed file.** A temporary file
carrying `{"u": [...]}` for a real label vector is scanned: **1 structural hit, 5
textual hits.** A clean control file in the same directory: **0 and 0.** Both are
asserted in `demo()`. The must-fire runs against a file this battery writes, so
its reachability cannot quietly stop depending on what happens to be in the tree
— which is the failure mode this round already catalogued once.

**The honest limit of the scan** is stated where it belongs, in §6: the textual
route finds exact decimal renderings only.

---

## 3. C-F(a) — THE do()-BIT EFFECT

### 3.1 delta, and the do(), both filed before anything was measured

**The corpus ships no bit.** `g` is `rng.uniform(0.0, 1.0, size=|B|)`
(`scale/r10_dual_oracle.py:325`). So the do()-bit is *defined*, in the module
header, before any effect was computed:

> `do(g_b := 1)` against `do(g_b := 0)`, so `Delta u = Omega[:, b]` where
> `Omega = (I - P_II)^-1 P_IB`.

`Omega[:, b]` is the absorption probability at `b` — the harmonic measure — so
the effect being measured is a quantity the repo's second oracle computes
independently, and `Omega.sum(axis=1) = 1` is available as an internal check
(measured `4.44e-16` … `2.22e-15`). **This is the largest single-coordinate
intervention the corpus admits**: any other `do(g_b := c)` scales `Omega[:, b]`
by `|c - g_b| <= 1`. The clause is given its best shot.

**`delta = 0.5`, and it was imported, not chosen.** It is `bar_verdict`'s shipped
one-sided do()-bit clause, `cal["flipper_dependence"] > 0.5`, at
`scale/negation_scope.py:1587` in the working tree and `:1563` at `bf2a769` —
same constant, the line moved. It is lifted out of that file's *text* at import
time rather than copied, so a drifted copy is impossible; `demo()` asserts
`DELTA == 0.5`. The statistic is that clause's own:
`flipper_dependence = mean|Delta y| / mean|y|`, the exact form asserted at
`tests/cameron/test_c1_propagate_registration.py:244`.

### 3.2 The measurement — every bit, not a chosen one

324 do()-bits across the 12 front-door rungs. The verdict is on the **weakest**
bit per rung, because a clause satisfied by the best available bit is satisfied
by construction.

| rung | `\|B\|` | `\|I\|` | `fd_min` | `fd_median` | `fd_max` | bits `>= 0.5` | `linf_min` | `linf_max` | whole-`B` flip |
|---|---|---|---|---|---|---|---|---|---|
| n64-d4 | 7 | 57 | `0.253141` | `0.353277` | `0.409953` | 0/7 | `0.344091` | `0.641376` | `0.339512` |
| n64-d6 | 4 | 60 | `0.359833` | `0.412808` | `0.585319` | **1/4** | `0.412845` | `0.497963` | `0.237044` |
| n128-d4 | 11 | 117 | `0.060288` | `0.153780` | `0.301723` | 0/11 | `0.196801` | `0.668976` | `0.221729` |
| n128-d6 | 7 | 121 | `0.110614` | `0.243077` | `0.416800` | 0/7 | `0.260875` | `0.545526` | `0.326109` |
| n256-d4 | 19 | 237 | `0.021741` | `0.106827` | `0.194100` | 0/19 | `0.175675` | `1.000000` | `0.168807` |
| n256-d6 | 16 | 240 | `0.042956` | `0.135818` | `0.222272` | 0/16 | `0.212528` | `0.556056` | `0.113851` |
| n256-d8 | 17 | 239 | `0.048212` | `0.105287` | `0.203569` | 0/17 | `0.153489` | `0.443958` | `0.074078` |
| n512-d4 | 41 | 471 | `0.007903` | `0.040661` | `0.098339` | 0/41 | `0.105632` | `1.000000` | `0.270344` |
| n512-d6 | 27 | 485 | `0.020849` | `0.072913` | `0.185670` | 0/27 | `0.240105` | `1.000000` | `0.193207` |
| n512-d8 | 31 | 481 | `0.017769` | `0.061104` | `0.116945` | 0/31 | `0.120985` | `0.547976` | `0.147573` |
| **n1024-d4** | 80 | 944 | **`0.000000`** | `0.022476` | `0.065246` | 0/80 | **`0.000000`** | `1.000000` | `0.188947` |
| n1024-d6 | 64 | 960 | `0.007056` | `0.033213` | `0.075733` | 0/64 | `0.156290` | `1.000000` | `0.185822` |

> **C-F(a): FAILS, 12 of 12. Filed `delta = 0.5`. Measured worst-case
> `flipper_dependence` `0.000000` … `0.359833`. 1 of 324 do()-bits in the whole
> corpus clears the filed delta** — one bit on `reproduce-n64-d6-t0.95`, at
> `0.585319`.

**The mechanism is arithmetic and it is not a near miss.** With `|B|` boundary
nodes the columns of `Omega` sum to one row-wise, so the average column carries
`1/|B|` of the label; `mean|u|` is about `0.4`. At `|B| = 80` the *typical* bit
therefore reads about `1/(80 x 0.4) = 0.031`, and the measured median is
`0.022476`. The clause is not failing by a factor of two, it is failing by a
factor of `|B|`, and it gets worse as the corpus gets bigger.

**`reproduce-n1024-d4-t0.95` reads exactly zero, and that is live data.** Seed
`0x3a14000a`, `|B| = 80`. One boundary node has **no interior neighbour** —
every node adjacent to it is also in `B` — so `Omega[:, b] = 0` identically and
its do()-bit moves the label by `0.0` in every norm. `spec.stratify` grows `B`
along a fixed node order (`scale/r10_corpus_spec.py:361-397`); nothing in that
rule forbids the order from putting a node and all its neighbours into `B`, and
at `|B| = 80` on a mean-degree-4 graph it eventually does. **So the corpus fails
even the campaign's weakest reading of the clause** —
`assert closed > 0.0, "a task declaring 0.0 must ship a do()-bit movement test"`
(`tests/cameron/test_c1_propagate_registration.py:247`) — on 1 of 324 bits.

**No other reading of the do() rescues it.** The whole-boundary intervention
`do(g := 1 - g)`, for which `Delta u = 1 - 2u` exactly by affinity of the
harmonic extension, reads `0.074078` … `0.339512` — also below `0.5` on 12 of 12.
And the norm question that bit it.14 §4 was checked here rather than assumed:
`linf` is carried beside `fd` in every row, and while `max_v |Delta u_v|` reaches
exactly `1.000000` on 5 rungs (an interior vertex of degree 1 hanging off a
boundary node absorbs there with probability one), the *weakest* bit's `linf`
still runs `0.000000` … `0.412845`, so `linf` does not clear `0.5` either.

### 3.3 C-F(a) shown able to fail, on a constructed input

The plant is a six-node graph written out in full — path `0-1-2-3-4` with
boundary node `5` hung off `0` alone, `B = {0, 4, 5}` — so its reachability
cannot depend on anything in the live corpus. `spec.admissible` **accepts** it,
which is the point: the guard that exists is not this one. Bit `5` has no
interior neighbour, `Omega[:, 5] = 0`, `fd_min = 0.000000`,
`bits_moving_nothing = 1`, verdict `failed`. `demo()` asserts all four.

The plant was written before rung 11 was measured. It then turned out to describe
a rung of the shipped corpus, which is the outcome a plant is for.

---

## 4. C-F(b) — L-SCOPE, AND IT IS THE HALF THAT MATTERS

### 4.1 The instrument is p1prime's, not a lookalike of it

`scale/p1prime.py` keys on **where** a plant enters, not who built it — P1 was
withdrawn for keying on the latter, and this report does not reintroduce that
error in either direction. Its production-builder vocabulary is a local inside
`main()` in the working tree and module-level at `bf2a769`, so it cannot be
imported. It is **lifted out of the parse tree** with `ast.walk`
(`scale/r10_it17_battery.py:_p1prime_builder`), which works in both layouts; the
regex is byte-identical between the two revisions and only moved.

The alias and front-door loops here mirror `scale/p1prime.py:104-115` and
`:118-129`, and a mirror is worth nothing unchecked, so it is **run against
p1prime's own shipped answers**:

```
"kind": "C-F-scope-binding", "pin": "06a180c", "tests_at_pin": 191,
"shipped_front_door": 46, "mine_front_door": 46, "disagreements": []
```

**191 of 191 agree, zero disagreements**, against `results/p1prime_front_door.txt`.
The census at HEAD, run through the same functions, reproduces the number this
iteration was handed:

```
"kind": "C-F-scope-census", "rev": "HEAD",
"files": 195, "front_door": 46, "no_front_door": 149
```

### 4.2 Where this round's controls actually enter

The production builder of a corpus instance is
**`dual.make_rung` (`scale/r10_dual_oracle.py:306`)**. It draws the graph, draws
the node order, runs `spec.stratify` — which runs `spec._lambda_at`, which runs
**`spec.admissible`** and `spec.spectrum` — and only then slices `B` and draws
`g`. That is the front door, and it is the only callable that emits an instance.

**`adm.recover` (`scale/r10_admissibility.py:178`) is not it.** It replays the
same rng stream with `|B|` supplied from a shipped row, which is exactly the
output of the stage it skips. It is a builder in the production package, so a
WHO-keyed test passes it; it enters **below** the door, so P1' does not.

| | `make_rung` | `recover` |
|---|---|---|
| `spec.draw_graph` | yes | yes |
| node-order permutation | yes | yes |
| **`spec.stratify` (`:361`)** | **yes** | **no** |
| **`spec._lambda_at` (`:352`)** | **yes** | **no** |
| **`spec.admissible` (`:162`, reached only via `:356`)** | **yes** | **no** |
| **`spec.spectrum` (`:222`)** | **yes** | **no** |
| **the `in_band` check (`r10_dual_oracle.py:329`)** | **yes** | **no** |
| `g` draw | yes | yes |

Both entries were run on the same 12 rungs and compared:
**`max_instance_gap = 0.0`** — the objects are identical, bit for bit. So the
below-door entry is not producing wrong instances here. **It is producing right
instances with five stages unexercised**, and the whole of it.16's battery, plus
the label set used for §2.2's scan, comes through it. That is the L-SCOPE
statement in its exact form: not "the input is wrong", but "the stages above the
entry point were never run, so nothing they would have caught was tested."

### 4.3 C-F(b) shown able to fail, twice, and the second one costs a verdict

**Instrument.** The classifier is run on the source of this battery's own two
control builders. `control_front_door`, whose body is `dual.make_rung(...)`, is
classified **front door**. `control_hand_built`, which writes an adjacency dict
out by hand, is classified **below the door** — and note that
`control_below_door`'s `adm.recover(...)` is *also* not a front-door call under
p1prime's own vocabulary, which is an independent confirmation of §4.2 rather
than an assertion of it. Both classifications are asserted in `demo()`.

**Consequence.** A star on 64 nodes — centre in `B`, 63 leaves of degree 1 — is
built by hand and pushed through the below-door entry. Every leaf absorbs at the
centre in one step, so:

| control | entry | `fd_min` | C-F(a) verdict |
|---|---|---|---|
| `reproduce-n128-d4-t0.95` | **front door** | `0.060288` | **failed** |
| hand-built star | **below the door** | **`2.000000`** | **ok** |

**A below-door control turns C-F(a) from `failed` into `ok` by a factor of 33.**
No production builder can emit that star: `lambda_2 = 0`, and `stratify` only
emits rungs inside `[0.90, 0.95]`. `demo()` asserts both verdicts, in both
directions — the clause firing on the front-door instance and *not* firing on the
star — because a scope check that fires on everything is worth as little as one
that fires on nothing.

---

## 5. COUNTS, SIX COLUMNS, SEPARATELY

`unscored` is the sixth: a row that carries no `status` at all — provenance, the
census, the two must-fire records — is counted as unscored rather than folded
into a pass. `not_reached` is SATURN's fifth category from it.15 and is carried
forward here for the one rung that was never generated.

| kind | passed | failed | errored | inapplicable | not reached | unscored |
|---|---|---|---|---|---|---|
| `C-D` | **12** | **1** (the plant) | 0 | 0 | 0 | 0 |
| `C-E-i` | **12** | 0 | 0 | 0 | 0 | 0 |
| `C-E-i-mustfire` | 0 | 0 | 0 | 0 | 0 | 1 |
| `C-E-ii` | **1** | 0 | 0 | 0 | 0 | 0 |
| `C-F-a` | **1** (the hand-built star) | **13** (12 rungs + the dead-bit plant) | 0 | 0 | 0 | 0 |
| `C-F-b` | 1 | 0 | 0 | 0 | 0 | 0 |
| `C-F-scope-binding` | 1 | 0 | 0 | 0 | 0 | 0 |
| `C-F-scope-census` | 0 | 0 | 0 | 0 | 0 | 1 |
| `C-A-carryover` | 0 | 0 | 0 | 0 | **1** | 0 |
| `provenance` | 0 | 0 | 0 | 0 | 0 | 1 |
| **total, 47 rows** | **28** | **14** | **0** | **0** | **1** | **3** |

The single `not_reached` is `widen-n128-d4-t0.905`, seed `0x3a150000`: it.15's
band miss, an instance that was never generated. It is carried into this battery
so that a corpus of 29 is not silently reported as a corpus of 28 that all
passed.

**Zero errored.** No clause raised on any instance.

---

## 6. THE AMENDMENTS, AND THEY ARE STATED NOT APPLIED

**C-F(a) is ill-posed as written, and it also fails under every reading of it.**
Those are two separate findings and both are reported.

> **Amendment (C-F).** *The clause must name three things it does not name: which
> intervention is the do()-bit, which statistic and norm the "effect" is measured
> in, and over which population of bits the bound must hold (the weakest, the
> median, or a declared draw).*

Under the readings filed here — the maximal single-coordinate flip, the
campaign's own `mean|Delta u| / mean|u|`, and the weakest bit — the corpus fails
`12 / 12` against the imported `delta = 0.5`. Under the *weakest* reading the
campaign uses anywhere, "the effect is strictly positive", it still fails, on 1
of 324 bits, and that failure is structural rather than statistical: nothing in
`spec.stratify` prevents a boundary node from having no interior neighbour. The
smallest repair — **not applied here** — is one clause in `spec.admissible`
refusing a boundary node all of whose neighbours are boundary, which would have
rejected `reproduce-n1024-d4-t0.95` at generation time.

**The imported delta is honest about where it came from, and about the mismatch.**
`0.5` was calibrated for `y = payload * sign`, where negating the flipper negates
the label and the exact ratio is `2.0`
(`scale/negation_scope.py:1570-1578`), and the campaign has already recorded once
that this band is the wrong shape for a label that spreads one driver's influence
over many positions
(`tests/cameron/test_c1_propagate_registration.py:250-262`). A harmonic label
spreads one boundary bit's influence over `|I|` positions and divides it among
`|B|` bits, so the mismatch here is larger still. **The number was still imported
rather than re-picked**, because a delta chosen after seeing `fd_median = 0.022`
would be a delta chosen to be cleared. If SATURN wants a delta for *this* corpus,
the amendment above is where it belongs and it should be filed before the next
measurement, not after this one.

**C-F(b) is well-posed and the corpus can satisfy it** — `make_rung` exists and
is the front door. The finding is that this round's own battery has not been
using it.

---

## 7. LIMITS

Collected here rather than scattered.

Only arm `reproduce` — 12 rungs — entered through the front door; that is 121
eigensolves and 26.2 s, and doing the same for the 16 `widen` rungs would have
cost another 165. The 28 label vectors used as needles in §2.2 were recovered
through `adm.recover`, the below-door path §4.2 is about; this is declared rather
than hidden, and it is bound at `recovery_max_range_delta = 0.0` against the
shipped fingerprints, so the objects are provably the shipped ones even though
the entry is not the front door. C-F(a) and C-D are measured only on the 12
front-door rungs.

The answer-key scan's textual route finds **exact decimal renderings only**: a
label stored at reduced precision, in a binary format, base64, rescaled, or split
across lines would pass it. The structural route covers the case that matters —
a whole label vector stored as JSON — and covers only `.json` and `.jsonl`. The
`2,089` undecodable files are counted, not read; a label key inside a `.npy` or a
`.pkl` would be in that 2,089. `lean/.lake` and `.git` are excluded by
declaration and their counts are printed; a key hidden in either would be missed,
and neither is a place this campaign writes.

`khop` is not a trained arm and no claim is made that a trained model reaches
`0.005092`; C-D asks whether the rig admits success, not whether a model achieves
it. The `k = 32`-to-`64` crossing is a property of NRMSE's denominator on this
label family and does not transfer to a corpus with a different mean-to-spread
ratio. Twelve rungs is not a sample size for anything but an existence claim and
a bound check.

The do()-bit is a *definition* made here, not one the corpus ships; a corpus that
later declares a different intervention will need this measurement redone. The
`fd` statistic uses `mean|u|` in its denominator, and `u >= 0` on this corpus, so
`mean|u| = mean(u)`; on a signed label the same statistic would behave
differently. `linf` and the whole-boundary flip are reported beside `fd` for
exactly the reason it.14 §4 gives, but four norms were not swept and only these
three readings were taken.

The front-door classifier agrees with p1prime on all 191 tests at its pin, which
binds it on that population; it has not been checked against a file that imports
`scale` through a mechanism neither of them models (a dynamic import, a `getattr`
chain). Both would be classified below the door, and both would be wrong.

No claim is made here about what a model can learn from this corpus. This
iteration adjudicates three admissibility clauses and nothing else.
