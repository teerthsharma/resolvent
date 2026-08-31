# The it.20 capacity line: which `d` S1 runs at, and whether 0.1748 is its constant

JUPITER, v-main.3M script iteration 20. The script line, verbatim:

> JUPITER: fetch F18-F21; SATURN builds Hungarian matcher `argmin_pi sum |‖k_i‖ − ‖k_pi(i)‖|`
> with residual sum/n printed per table; capacity line: role coherence floor = 0 for
> `k <= d` (Welch), random-role coherence ~ 0.1748 at `d=256, k=16` (corrected constant,
> MC CI [0.17446, 0.17513]) — the scramble control's expected residual, not zero.

Code `scale/r10_it20_coherence.py`, rows `results/r10_it20_coherence.jsonl` (12 rows).
Reproduce every number below with

```
python -m scale.r10_it20_coherence            # table + assert-based demo(), ~70 s
python -m scale.r10_it20_coherence --write    # same, and writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, scipy 1.17.1, torch 2.5.1+cu121, Windows 11 x86-64,
branch `feat/r9-causal-consequence` at `bf2a769`. Seeds declared **before** the run
and pinned in the module header: Monte Carlo `seed=0`, `trials=20000` (deliberately
the same cell `scale/coherence_floor.py` publishes, so the `d=256` row is a
reproduction and not a re-pick); eval batch `seed=12345`; arm `seed=0`; bootstrap
`seed=0`. Nothing reads a clock, a hash seed or a thread count; re-run and the
`.jsonl` is byte-identical.

`scale.vram_gate.preflight(0, name='r10-it20-coherence', host_mib=512)` returned
`HOST FITS -- needs 640 MiB, 4894 MiB available` before the run. One process, no
training, no GPU. Peak is one `(2000, 128, 16)` float64 Monte Carlo block and one
`[512, 64, 16]` float32 eval batch.

---

## 0. THE HEADLINE

**S1 runs at `d = 16`, `k = 8`. The cited constant is correct for `d = 256, k = 16`
and does not travel to that configuration.** Three separate failures, of decreasing
size, and the first is fatal on its own:

1. **Wrong quantity.** The matcher's residual is `|‖k_i‖ − ‖k_π(i)‖|` — a difference
   of key *norms*, in key-norm units. Coherence is a dimensionless cosine. No value
   of the second is the expected value of the first. §5.
2. **Wrong dimension.** At `d = 16, k = 8` the random-unit-vector max coherence is
   **`0.547180` [0.545987, 0.548373]**, `3.130×` the cited `0.1748`, CIs disjoint by
   `0.371`. At the *real* selected key directions it is **`0.719784`**, `4.12×`. §4.
3. **The `k <= d` premise fails at two of the three shipped `k`.** `k = 32` and
   `k = 128` are the `--ks` default of eight sweep scripts; at `d = 16` the Welch
   bound there is `0.179605` and `0.234772`, not zero. §3.

**The correct number for the sentence the rule is trying to write:** the scramble
control's expected residual at the shipped configuration is
**`0.157792` [0.148557, 0.167109]** through the shipped producer, corroborated at
**`0.149191` [0.140212, 0.158170]** by an independent fresh-permutation path — in
key-norm units, `1.75–1.85×` the key-norm sd. §6.

**The rule's *conclusion* survives all three.** The scramble control's expected
residual is emphatically not zero; a control calibrated to zero is vacuous. Only the
constant attached to that conclusion is wrong, and it is wrong in the direction that
*understates* the residual's significance relative to the spread. The smallest
amendment is in §7. **This is not a strike against the rule's reasoning — it is
MISTAKES.md V-17 applied to the rule's own arithmetic, by the rule's own logic.**

---

## 1. F18–F21 — WHAT THE ITERATION RESTS ON

"Fetch" here means what it meant at it.14: **located in this repo and cited by file
and line, or declared absent and named.** Nothing was fetched from the internet.

**The F-numbering in this project collides across three namespaces, and this is
stated rather than resolved by guess.** `results/r10_it16_admissibility.md:225-228`
uses `F0–F3` for feature families; `results/r10_it14_corpus_spec.md:60-67` uses
`F1–F8` for that iteration's mathematical prerequisites; `PHASE2_CONTRACT_V_MAIN_4.md:69`
heads a table **"New fetches"** (rows at `:74-83`) that opens at `F24`, implying a
running series through `F23`. **No `F9–F23` table exists anywhere in this tree** — searched 129 `.md` and
374 `.py` files outside `.claude/worktrees`. The script line as delivered does not
state F18–F21's subjects.

So both readings are given, and neither is invented.

| | Reading A — the repo's own F-register | Status |
|---|---|---|
| **F18** | **Routing beats softmax, and the comparison survives F16**, because `pivot_unsigned` was always the non-negative arm. softmax `0.877168` [0.830455, 0.924226] vs pivot_unsigned `0.747528` [0.696849, 0.797716], disjoint, 4769 params each. | **CITED.** `CHECKLIST.md:368-370`; `LOOP_PROMPT_ROUND4_ARCHIVE.md:50-53`. |
| **F19** | **The budget is the bar.** M3 readings below `n_train=8192` rank overfitting: `128 → 2.116579`, `512 → 1.316514`, `2048 → 0.949529`, `8192 → 0.877168`. | **CITED.** `CHECKLIST.md:372-373`; `LOOP_PROMPT_ROUND4_ARCHIVE.md:55-58`. |
| **F20** | **Co-prime dilation, and G1 fired on it.** `[1,3,5,7]` cuts severance `0.5745 → 0.1277` at `s=128`; the mechanism is PRIOR ART (arXiv 2606.28560's "coprime (anti-gridding) reassignment"), so arm A proceeds as the difference-set coverage theorem and never as "co-prime spacing". | **CITED.** `CHECKLIST.md:375`; `LOOP_PROMPT_ROUND4_ARCHIVE.md:60-63`. |
| **F21** | **The batched path is free.** `batched_pivot_hop2` / `batched_select_pivots` are bitwise equal to the loop (`torch.equal`, 12 forward cases, gradient maxdiff `0.0` at `n=8`) and `374×` faster on forward+backward at `n=2048`. Declared limit: gradients bound only to `n ≤ 64`. | **CITED, and only in one place.** `LOOP_PROMPT_ROUND4_ARCHIVE.md:65-69`. **Not in `CHECKLIST.md`** — the register carries F18/F19/F20 at `:368/:372/:375` and stops. |

**Reading B — a running fetch series continuing `F1–F8` (it.14) and preceding
`F24–F28` (Phase 2): UNRESOLVED, and no subject is assumed for any of the four.**
No line was invented, exactly as for F8 at it.14.

**F21 is directly load-bearing for this iteration and F18–F20 are not.** The it.20
matcher runs on `batched_select_pivots`' output (`scale/fgreen_matched.py:63`,
`SHIPPED_SELECT = m3.batched_select_pivots`), so F21's bitwise-equality bind is what
lets §4 and §6 read the batched selector's pivots as the loop's. **Its declared limit
does not bite here**: the bind is forward-only above `n = 64`, and every reading in
this report is a forward pass on an untrained arm.

### The results this iteration actually rests on, named regardless of number

| | Result | Status |
|---|---|---|
| **W** | **The Welch bound.** `max_{i<j} \|⟨u_i,u_j⟩\| ≥ sqrt((k−d)/(d(k−1)))` for any `k` unit vectors in `R^d`. | **CITED.** `scale/coherence_floor.py:66-71`; `MATHEMATICS.md:812-821`. |
| **C** | **The corrected random-coherence constant, `0.174795` at `d=256, k=16`, MC CI `[0.174460, 0.175131]`**, corroborated `0.174499` by order-statistic quadrature; the author's `sqrt(2 ln k / d) = 0.147176` is 15.8% low because the union bound runs over `k` events rather than `C(k,2) = 120` pairs. | **CITED, and imported rather than re-picked.** `MATHEMATICS.md:824-848`; `scale/coherence_floor.py:18-47` (the published cell), `:102-133` (`monte_carlo`). Origin: `.superpowers/sdd/polymorphic-drifting-squirrel/progress.md:451-460`. |
| **M** | **The Monge closed form as the matcher's free oracle.** Cost `\|a_i − b_j\|` on the real line satisfies the Monge condition, so sorting both pools and pairing rank-for-rank is provably optimal in `O(n log n)`; computed alongside the LSA solver and raising on disagreement. | **CITED.** `scale/matcher.py:26-40` (derivation), `:153-192` (`monge_optimum`, `match`). |
| **G** | **The shipped geometry.** `D_MODEL = 16`, `K_PIVOTS = 8`, keys are `nn.Linear(d_model, d_model)` applied to `[n, s, d_model]`. | **CITED.** `scale/m3_capability.py:79`, `:87`, `:109`. |

---

## 2. WHICH `d` AND `k` S1 ACTUALLY RUNS AT — MEASURED, NOT ASSUMED

The it.20 construction is not hypothetical; it is already built. **`scale/fgreen_matched.py:114-140`
(`imbalance_table`) is exactly the rule's matcher**: the causal pool is the shipped
top-`k` pivots' key norms, each control's pivots are the filler pool, the assignment
is `matcher.match`, and the printed number is `Σ|Δnorm|/n`. Its scramble control is
`select_random` at `:86`.

The vector whose coherence the capacity line speaks about is therefore the **key
vector** `k_i = wk(x_i)`, and its dimension is fixed in one place:

```
scale/m3_capability.py:79    D_MODEL = 16          # fixed per task spec
scale/m3_capability.py:87    K_PIVOTS = 8          # pivot count, matches pivot_probe's default
scale/m3_capability.py:109   self.wk = nn.Linear(d_model, d_model, bias=False)
```

`wk : R^16 → R^16`, so **a key vector lives in `R^16`**. Confirmed executably rather
than by reading: the eval tensor is `torch.Size([512, 64, 16])` and
`real_keys().shape[-1] == 16` is asserted in `demo()`.

### The `d = 256` that exists in this repo is a different `d`

`scale/foreman_consequence.py:4` documents `--s 512 --d 256`, and
`results/foreman_consequence_d256.txt:1` reads `s=512 d=256`. **That `d` is the task
width, not the vector dimension.** The same call site passes both:

```
scale/foreman_consequence.py:427   xe, ye, f, p = make_batch(n_eval, s, d, d_model=SF.DMODEL, seed=seed + 12345)
scale/foreman_signfloor.py:82      N, S, D, DMODEL, SEED = 256, 64, 24, 16, 0
```

`d = 256` and `d_model = 16` in the same call. **The `d=256` consequence run puts key
vectors in `R^16`, not `R^256`.** Note also `256` appears there as `N`, the draw
count, at `:82` — three distinct quantities wearing the number 256 within one module.

### Census, so the claim is a count and not an impression

Every numeric `d_model` / `D_MODEL` / `DMODEL` / `n_embd` literal in all **374 `.py`
files** outside `.claude/worktrees`:

| value | count | where |
|---|---|---|
| **16** | **31** | the M3 / capacity / consequence / e-task path throughout |
| 32 | 2 | `ceq/arms.py:33`; `tests/cameron/test_identity_manifest.py:178` |
| 1024 / 1280 / 2048 | 3 | `ceq/sizing.py:111,114,122` — `CFG_300M/500M/1B`, projection configs |
| **256** | **0** | — |

**Nothing in this repository puts a key or role vector in 256 dimensions.**
`tests/gpu/test_cuda_parity.py:37` states the same conclusion independently: *"model
width is M3.D_MODEL=16 everywhere, exactly as the journalled units."*

### Which `k`

`K_PIVOTS = 8` is the shipped default and the value `fgreen_matched` prints
(`results/fgreen_matched_it1.txt:8`, `k=8`). But `k` is swept: `[8, 32, 128]` is the
`--ks` default in **eight** modules — `aggregator_matched_filler.py:155`,
`aggregator_mechanism.py:161`, `arm_s.py:485`, `b1_collapse_test.py:154`,
`b1_decomposition.py:153`, `chase_slope_ci.py:125`, `foreman_hilbert.py:461`,
`foreman_theta_tv.py:135` (plus `foreman_curvature.py:41`, `foreman_quantisation.py:55`
as literals). **`(d, k) = (16, 8)` for the headline cell; `k ∈ {8, 32, 128}` across the
sweep.** Both matter, for different reasons, in §3 and §4.

---

## 3. THE WELCH FLOOR — TRUE, VACUOUS WHERE IT APPLIES, AND FALSE AT TWO SHIPPED `k`

**The bound.** For any `k` unit vectors in `R^d`,

```
    max_{i<j} |⟨u_i, u_j⟩|  ≥  sqrt( (k − d) / (d (k − 1)) )
```

For `k ≤ d` the numerator is `≤ 0`, so the bound is `0`, and it is **attained**: `k ≤ d`
orthonormal vectors exist and have all pairwise inner products exactly zero. So for
`k ≤ d` the bound is not merely zero but **tight and vacuous** — it says nothing that
`|⟨u_i,u_j⟩| ≥ 0` does not already say. **In that regime the Welch bound constrains
nothing whatsoever, and a capacity line that quotes it is quoting an identity.**

**The bound is real above the dimension**, which is what makes the instrument
non-trivial and is asserted in `demo()`: at `d = 2, k = 3` the bound reads `0.5`, and
three unit vectors at 120° attain `|cos 120°| = 0.5`, agreeing to `1e-12`.

**Measured, at every configuration this repo runs** (`scale/r10_it20_coherence.py`,
`kind="welch"` rows):

| `d` | `k` | `k ≤ d` | Welch | reading |
|---|---|---|---|---|
| 256 | 16 | yes | `0.000000` | **VACUOUS** — the cited configuration |
| **16** | **8** | **yes** | **`0.000000`** | **VACUOUS** — S1's headline cell |
| 16 | 16 | yes | `0.000000` | VACUOUS — the boundary |
| **16** | **32** | **no** | **`0.179605`** | **BINDS** |
| **16** | **128** | **no** | **`0.234772`** | **BINDS** |

**The rule's Welch clause is true at S1's headline cell and false at two of the three
shipped `k`.** At `k = 32` and `k = 128` in `d = 16` there is a genuine, positive,
unavoidable floor. A capacity table that prints "floor = 0" beside a `k = 32` or
`k = 128` row is printing something the geometry forbids.

**And there is a trap in that column worth naming.** `welch(16, 32) = 0.179605` is
within 2.7% of the cited `0.1748`. A reader who saw `0.1796` in a `k=32` row and
`0.1748` in the capacity line would very reasonably conclude the two agreed. They are
different quantities at different configurations and their near-equality is
arithmetic coincidence.

### Does the bound apply to S1's construction at all?

**Only after a normalisation S1 does not perform.** The Welch bound and every
coherence figure in the rule are statements about **unit** vectors. S1's keys are not
unit vectors — measured, `‖k_i‖` has mean `0.227332`, sd `0.085214`, range
`[0.061524, 1.060215]` over `64 × 64` keys. **The entire reason the it.20 matcher
exists is that those norms vary** (`scale/matcher.py:3-8`: the selector ranks by
`key.norm(dim=-1)`, so high key-norm and causal-arm membership are confounded by
construction).

So the coherence claim applies to the **directions** `k_i/‖k_i‖`, and the matcher
residual is about the **magnitudes** `‖k_i‖`. They are the two orthogonal halves of
the same vectors, and the rule identifies one with the other. §5.

---

## 4. THE RANDOM-ROLE CONSTANT, MEASURED AT BOTH DIMENSIONS

Monte Carlo, `k` fresh normalised-Gaussian unit vectors in `R^d`, 20 000 independent
trials, **seed 0 declared before the run**, using `scale/coherence_floor.monte_carlo`
unchanged. Reported statistic: sample mean over trials of `max_{i<j} |⟨u_i,u_j⟩|`,
normal 95% CI on that mean.

| `d` | `k` | MC max | 95% CI | ÷ 0.1748 | is 0.1748 in the CI? |
|---|---|---|---|---|---|
| **256** | **16** | **`0.174795`** | **[0.174460, 0.175131]** | `1.000×` | **YES** |
| **16** | **8** | **`0.547180`** | **[0.545987, 0.548373]** | **`3.130×`** | **NO** |
| 16 | 32 | `0.713405` | [0.712706, 0.714104] | `4.081×` | NO |
| 16 | 128 | `0.811148` | [0.810703, 0.811593] | `4.640×` | NO |

**The cited constant reproduces exactly at the configuration it was derived for.**
`0.174795`, CI `[0.174460, 0.175131]` — the rule's `~0.1748` and `[0.17446, 0.17513]`
are that cell to the digits quoted, and it matches `MATHEMATICS.md:833-834` and
`scale/coherence_floor.py:27-28` line for line. **The constant is not wrong and was
not re-picked here; it was imported and reproduced.** The rule's parenthetical
"corrected constant" is also correct: the author's `sqrt(2 ln 16 / 256) = 0.147176`
is 15.8% low, and `MATHEMATICS.md:826-841` gives the reason (union bound over `k = 16`
events rather than `C(16,2) = 120` pairs) with an independent quadrature corroboration
at `0.174499`.

**It simply does not describe `d = 16`.** The two intervals are disjoint by `0.371` —
1 105 half-widths of the `d=256` interval, 311 of the `d=16` one. Coherence scales roughly as `d^{-1/2}`, and
`256/16 = 16`, so a factor near `4` before the `k` correction is exactly what the
geometry demands; the measured `3.13×` is that, damped by `k` falling `16 → 8`.

### And the random-unit-vector model itself understates the real construction

The model assumes independent uniform directions. S1's keys are one linear map applied
to correlated token features, so whether the model applies is a claim. Measured on the
shipped top-8 key directions, 64 draws, arm seed 0, eval seed 12345:

| quantity | `d=16, k=8` random unit vectors | **shipped top-8 key directions** |
|---|---|---|
| max pairwise `\|cos\|` | `0.547180` [0.545987, 0.548373] | **`0.719784` [0.698242, 0.741325]** |
| mean pairwise `\|cos\|` | `0.203038` | **`0.272342` [0.263190, 0.281494]** |

**The real construction is *more* coherent than random, not less** — CIs disjoint —
because real keys are correlated. Against the actual object, the cited `0.1748` is
**`4.12×` low**. (For scale: all 64 tokens' directions carry max `|cos| = 0.879994`
[0.872853, 0.887134].)

**The direction of the error is the safe one, and that is worth stating plainly.**
Every one of these numbers is *larger* than `0.1748`, so the rule's conclusion — that
a scramble control expecting zero is vacuous — is strengthened, not weakened, by the
correction. The rule under-states its own case by a factor of three to four.

---

## 5. THE QUANTITY ERROR, WHICH IS FATAL INDEPENDENTLY OF THE DIMENSION

Even at `d = 256, k = 16`, `0.1748` **would not be the matcher's expected residual**,
because the two are not measurements of the same thing.

| | the matcher's residual | coherence |
|---|---|---|
| formula | `(1/n) Σ_i \|‖k_i‖ − ‖k_π(i)‖\|` | `max_{i<j} \|⟨u_i, u_j⟩\|` |
| operates on | **magnitudes** `‖k_i‖` | **directions** `k_i/‖k_i‖` |
| units | key-norm units — rescale the keys and it rescales | dimensionless, scale-invariant |
| range | `[0, ∞)` | `[0, 1]` |
| defined for | any two pools of `n` reals | unit vectors only |

The residual is **not scale-invariant**: multiply every key by 10 and it multiplies by
10, while every coherence figure is unchanged. A constant with units cannot be
imported into a slot that has none, and `‖k_i‖` at the shipped geometry has mean
`0.227332` — so `0.1748` read as a residual is silently asserting a scale, and the
shipped scale is one where `0.1748` would be 77% of the mean key norm.

**Where the conflation entered, traced.** `MATHEMATICS.md:849-853` says: *"Scrambled
roles at `d = 256`, `k = 16` carry an expected mean overlap of `0.049917` and an
expected worst-case overlap of `0.174795` … A scramble control calibrated to expect
zero residual separation is vacuous before it runs."* The source is careful — it says
**overlap**, and uses "residual separation" for a *different* clause of the argument.
The script line compresses two sentences into an apposition — `0.1748` *"— the
scramble control's expected residual"* — and the compression is where a cosine became
a residual. **The source did not make this error.** This is inheritance across a
paraphrase, which is why re-deriving from a cited constant is not the same as
re-reading the sentence around it.

### 0.1748 in this repository names four unrelated quantities

Round law: import constants that exist, **and state which quantity the source
measured**. Four distinct quantities in this tree sit within 15% of `0.175`:

| value | quantity | source |
|---|---|---|
| `0.174795` | MC max pairwise coherence, `d=256, k=16` | `MATHEMATICS.md:833`; `scale/coherence_floor.py:27` |
| `0.17480` | **content-conditional sign rate at `s=8`** — nothing to do with coherence | `ceq/hf/modeling_ceq.py:132`; F6 at `house-events.jsonl:686` |
| `0.179605` | **Welch bound at `d=16, k=32`** — a floor, not a mean | measured here, §3 |
| `0.157792` | **the scramble control's actual residual** | `results/fgreen_matched_it1.txt:11` |

A fifth, `1.749951e-01`, is the smallest gap a planted degree defect produces in the
Kirchhoff oracle (`results/r10_it14_corpus_spec.md:65`). **A number near 0.175 in this
codebase carries essentially no identifying information**, and any cell quoting one
must name its quantity or it cannot be checked.

---

## 6. WHAT THE SCRAMBLE CONTROL SHOULD ACTUALLY EXPECT

This is the number an S1 table is read against, so it is measured on the shipped path
rather than modelled.

**Through the shipped producer** — `fgreen_matched.imbalance_table`, 64 draws, `k=8`,
eval seed 12345, arm seed 0, called in-process so the figure is reproduced and not
copied out of a text file:

| control | residual | 95% CI | reading |
|---|---|---|---|
| **`pivot_random`** (the scramble) | **`0.157792`** | **[0.148557, 0.167109]** | **the expected residual** |
| `pivot_band` (the matched control) | `0.108689` | [0.101061, 0.116511] | for scale |

`0.157792` reproduces the published cell at `results/fgreen_matched_it1.txt:11` to
better than `5e-7`, asserted in `demo()`.

**Second path, independently.** The shipped `select_random` re-seeds its generator
inside every call and `imbalance_table` calls it one draw at a time, so the shipped
control applies **one fixed permutation** to all 64 draws — that is one scramble
measured 64 times, not 64 scrambles. A fresh permutation per draw gives the population
expectation: **`0.149191` [0.140212, 0.158170]**. The fresh-permutation mean lies
inside the shipped interval and vice versa; **the two paths agree**, and the agreement
is asserted in `demo()`. (This is a property of the shipped control, noted, not a
defect claim: a fixed pivot set across forward passes is deliberate, per
`scale/fgreen_matched.py:87-92`.)

**Against the spread it failed to remove**, which is the only scale on which a
residual is readable (`scale/matcher.py:49-52`): key norms have mean `0.227332`,
sd `0.085214`, so

* shipped scramble residual = **`1.852 ×` sd** = 69.4% of the mean key norm
* fresh-permutation residual = **`1.751 ×` sd**
* band (matched) residual = `1.276 ×` sd — the figure already on record at
  `CHECKLIST.md:888`, which this reproduction recovers to 5 significant digits

**The rule's point stands and is understated.** The scramble control's expected
residual is not zero; it is `0.157792`, and it is **larger than the population sd of
the very quantity it was supposed to balance**. A scramble control calibrated to zero
is not merely vacuous — at this geometry it is off by nearly two standard deviations.
The rule reached the right conclusion with the wrong number.

---

## 7. THE VERDICT, AND THE SMALLEST AMENDMENT

**The rule is ill-posed as written for the configuration S1 runs at.** Reported, not
silently repaired, per the round law.

| clause | verdict |
|---|---|
| "role coherence floor = 0 for `k <= d` (Welch)" | **TRUE, and vacuous where it holds.** Holds at `(16, 8)`. **FAILS at `(16, 32)` and `(16, 128)`**, two of the three shipped `k`. |
| "random-role coherence ~ 0.1748 at `d=256, k=16`, MC CI [0.17446, 0.17513]" | **TRUE and reproduced exactly.** Also correctly flagged as the corrected constant. |
| "…at `d=256, k=16`" as S1's configuration | **FALSE.** S1 runs `d = 16, k = 8`. 0 of 36 `d_model` literals in 374 `.py` files are 256. |
| "— the scramble control's expected residual, not zero" | **The "not zero" is RIGHT and is the rule's real content. The number attached to it is WRONG twice**: wrong quantity (cosine vs key-norm units) and wrong dimension. |

### The smallest amendment

Split the one sentence into the two claims it is carrying, and anchor each to its own
configuration and quantity:

> **capacity line:** role coherence floor `= sqrt((k−d)/(d(k−1)))`, which is `0` and
> vacuous at `d_model = 16, k = 8` and is `0.179605` / `0.234772` at `k = 32` / `128`
> — print the formula per row, not the constant. Random-unit-vector max coherence at
> the shipped `d_model = 16, k = 8` is **`0.547180`** [0.545987, 0.548373], MC seed 0,
> 20 000 trials; the shipped top-8 key **directions** carry **`0.719784`**
> [0.698242, 0.741325]. **Separately**, the scramble control's expected matcher
> residual — key-norm units, not a coherence — is **`0.157792`** [0.148557, 0.167109],
> `1.85 ×` the key-norm sd. Neither is zero, and a control calibrated to zero is
> vacuous.

Three edits: quote the Welch **formula** per row instead of a constant; move the
`0.1748` cell to where it belongs (a `d = 256` illustration, if it is wanted at all)
and replace it with the measured `d = 16` figure; and separate the coherence sentence
from the residual sentence, because they are different quantities.

**Cost if this amendment is wrong**, stated as it.16 requires: if S1 is in fact a
separate experiment at `d = 256` that no code in this tree implements yet, then the
capacity line is fine as written and §4's `d=16` column is the wrong comparison — but
the quantity error in §5 survives that, and so does the `k ≤ d` failure at `k = 32`
and `k = 128`, because those are the shipped sweep values under either reading.

---

## 8. SHOWING THE INSTRUMENTS CAN FAIL, BEFORE TRUSTING THEM

Round law. All four are `assert`s in `demo()`; each would pass trivially if the
instrument were blind.

1. **The Welch bound is not identically zero.** `welch_bound(2, 3) = 0.5` to `1e-12`,
   and three explicitly-constructed unit vectors at 120° attain `|cos| = 0.5` to
   `1e-12`. A bound that were always `0` would satisfy every other assertion here and
   mean nothing.
2. **The Monte Carlo is sensitive to the sampler.** Drop the normalisation and the
   quantity stops being a cosine: the un-normalised mean must fall **outside** the CI
   the normalised sampler produces. Asserted as a `not (lo <= x <= hi)`.
3. **The matcher answers zero only when it should.** Identical pools give residual
   **exactly** `0.0`; a pool shifted by `0.75` gives back `0.75` to `1e-12`. Its own
   Monge oracle is verified against the LSA solver on the first draw of every run
   (`verify=(i == 0)`).
4. **The must-fire for this iteration's actual claim.** The `d=16, k=8` CI must be
   **disjoint from and above** the `d=256, k=16` CI, and the ratio must exceed `3.0`.
   If a coherence estimator returned the same number at both dimensions, that
   assertion is what catches it.

### Counted, alongside what was found

| examined | count | found |
|---|---|---|
| `.py` files searched for `d_model` declarations | **374** | 36 numeric literals; **31 are `16`**, **0 are `256`** |
| `.md` files searched for `F18`–`F21` and for an `F9`–`F23` table | **129** | F18/F19/F20 in `CHECKLIST.md`; F18–F21 in `LOOP_PROMPT_ROUND4_ARCHIVE.md`; **no `F9`–`F23` table** |
| modules whose `--ks` default is `[8, 32, 128]` | **8** | 2 of the 3 values put `k > d` at `d_model = 16` |
| `(d, k)` configurations Monte-Carlo'd | **4** | 1 covers `0.1748`; 3 do not |
| scramble-control paths measured | **2** | agree; `0.157792` and `0.149191` |
| repo constants within 15% of `0.175` | **5** | 5 different quantities |
| rows written to `results/r10_it20_coherence.jsonl` | **12** | — |

---

## Limits

The coherence figures for the real key directions are on an **untrained** arm
(`torch.manual_seed(0)`, `m3.Arm("pivot_unsigned", 64)`), because a capacity floor is
a statement about what the geometry permits before training; trained keys may be more
or less coherent and are not measured here. The `0.719784` figure is 64 draws, not
20 000, and its CI is correspondingly wide. `demo()`'s Reading-B verdict is a negative
search result over this tree, not proof that no `F9`–`F23` table exists elsewhere —
if the window holds one, its F18–F21 subjects supersede §1's Reading A. The `d=256`
Monte Carlo is a reproduction of `scale/coherence_floor.py`'s own published cell at
the same seed and trial count, so it corroborates the arithmetic and not the seed
choice. Whether S1 is a separate `d = 256` experiment cannot be settled from code that
does not exist; §2's census settles only that no such code is in this tree today.
