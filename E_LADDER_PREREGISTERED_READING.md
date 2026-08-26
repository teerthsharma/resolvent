# Ladder E — the settled−twin dose-response, pre-registered before the run

Round 7, iteration 30 window. Owner: Chase. Written **before `scale/m3_quintuple.py`
has produced a single number on any `e3_t*` task**. The pilot SD in section 4 and
the fallback product in section 7 are computed from the *already-published*
`negation_scope` journal (`results/m3_quintuple_v2.jsonl`, 25 units, commit
`9593f4a`), which is the only data that existed when this file was written.

Instrument: `scale/m3_quintuple.py --task e3_t{1,2,8,32}`, the `--task` flag ported
from `scale/m3_capability.py:247`. Corpus: `scale/negation_scope.py::M3_TASKS`,
registered by Cameron. Statistics rule: `LOOP_PROMPT.md` §1.8.

---

## 1. The one contrast, and why every other one on this ladder is void

`LOOP_PROMPT.md` §1.7d, verbatim:

> **E3 shares E1's oracle.** `M3_TASKS` registers every `e3_t{t}` with
> `equilibrium_oracle` — the same signed path sum for which E1 is declared a
> rigged demo (`scale/negation_scope.py:635` and `:640`). The settled arm's own
> resolvent computes that object, so **`settled − softmax` on any e3 task is
> credited nothing**: it inherits E1's rig at every chain length.

Therefore, on `e3_t*`:

| contrast | status on this ladder |
|---|---|
| `settled − twin` | **the only creditable contrast.** Identical mixture, identical parameter tensors, differs only in the iteration. |
| `settled − softmax` | **VOID — inherits E1's rig.** Not run, not printed as a verdict. |
| `twin − softmax` | not creditable here either; the twin is built on the same resolvent geometry. Read on `negation_scope` only, where it already is (§7). |
| `settled − argmax` | not run on this ladder. It answered the lopsidedness question on `negation_scope` and answers nothing about `t*`. |

Every `e3` row printed by this run carries the rig caveat beside it.

## 2. The shape that would be believed

`LOOP_PROMPT.md` §1.7d: *"The believable signature is dose-response: the contrast
compatible with zero at `t* ≤ 1` and growing with `t*`. Flat in `t*` ⇒
routing-only (K-2E). Nonzero at `t* ≤ 1` ⇒ capacity leakage (G4 shape)."*

`t*` is the exact hop count of the label (`scale/negation_scope.py::make_equilibrium_batch`
zeroes the sub-diagonal at and before `head = s-1-t*`, so `A` is nilpotent of index
`t*+1` on the read coordinate). The arms' own hop budget is 1 (softmax) or 2 (pivot
family); the settled cell settles the pivot **weights**, not the token chain, so it
too reads a two-hop neighbourhood. `t* = 1` is therefore inside every arm's budget
and `t* = 32` is outside all of them.

## 3. The cells and the geometry, fixed here

```
cells    settled, twin          (2)
rungs    e3_t1, e3_t2, e3_t8, e3_t32   (4)
seeds    0 1 2 3 4              (5)
s=64  d=24  steps=150  n_train=2048  n_eval=2048  k_piv=8
beta=0.5  t_max=21  n_neumann=21  threads=2  cpu
```

40 units. **This geometry is chosen to be byte-identical to the geometry
`scale/etask_k5e.py` is running right now** (`--n-train 2048 --n-eval 2048
--steps 150 --seeds 0 1`), so that this run's `settled` and `twin` values at seeds
0 and 1 must reproduce that file's rows exactly. Both route through
`paired_arm.train_and_predict` with the same `batch_fn` and the same seed, so the
match is required to be **bitwise**, and a mismatch is an instrument failure, not
a disagreement. That cross-file bind is claimed here in advance so it cannot be
discovered afterwards as a coincidence.

## 4. What was dropped, and the arithmetic that dropped it

Measured unit cost, from `results/m3_quintuple_v2.jsonl`'s own `meta.seconds`
at `s=64 d=24 steps=150 n_train=8192 n_eval=512 k=8`, 2 threads, this box:

```
softmax  mean 106.2 s   min 101.6   max 116.0
glance   mean 162.0 s   min 123.3   max 211.4
settled  mean 330.2 s   min 298.6   max 374.8
twin     mean 513.3 s   min 366.6   max 779.2
argmax   mean 225.4 s   min 196.7   max 240.6
                        total for 25 units: 6685.3 s
```

The cross the brief names — 3 arms × 5 tasks × 5 seeds = 75 units — prices at
roughly `25 × (330.2 + 513.3 + 106.2) = 23 738 s ≈ 6.6 h` at `n_train = 8192`, on
a box currently running four other measurement processes. It does not fit in one
iteration and it is not attempted.

**Dropped, each with its reason:**

1. **`n_train` 8192 → 2048.** Cuts the training term by 4×. Justified because no
   `e3` reading is published at any `n_train`, so nothing is disturbed; and
   because it buys the §3 cross-file bind against `etask_k5e.py` for free.
   *Cost of the drop:* the pilot SD in §5 was measured at 8192 and the SD at 2048
   is unknown and probably larger. Recorded as a limit, not as a result.
2. **`softmax`, `glance`, `argmax` cells.** `settled − softmax` is void on this
   ladder (§1); `glance` is bitwise `softmax`; `argmax` answers the lopsidedness
   question, which is not a `t*` question. `softmax` rows at this exact geometry
   come free from `results/etask_k5e.txt`. Saves 35 units.
3. **Seeds 13 → 5.** §1.8 permits exactly one pre-registered fixed-sample read at
   an `N` fixed by power analysis before the data land. `N = 5` is that `N`, and
   §5 states what it can and cannot see. 13 seeds would be 104 units.

**What is NOT dropped:** all four rungs. A two-point ladder cannot distinguish
"growing with `t*`" from "a step at one rung", and growth is the whole claim.

**Rung order, fixed here because the box is contended and the run may not
finish.** At the moment of writing, `Get-CimInstance Win32_Process` shows ten
other Python measurement processes on this box, each pinned to 2 threads of 28
logical processors; four of them (`foreman_consequence`, `foreman_signfloor`)
have been running 30 minutes. Wall-clock per unit will therefore exceed the
journalled figures above by an unknown factor. The rungs run in the order

```
e3_t1   ->   e3_t32   ->   e3_t8   ->   e3_t2
```

— **the two endpoints first**, because they carry the direction (`t* = 1` is the
"compatible with zero" prediction, `t* = 32` is the "grows with `t*`" prediction)
and the two interior rungs only distinguish outcome **A** from outcome **B**. A
rung that does not complete is reported as **NOT RUN** and its cell is left
blank. It is never reported as a null, and no verdict row in §6 may be claimed
on a ladder whose missing rung could have changed it: rows **A**, **C** and **F**
all quantify over *every* rung and are unavailable until every rung is in. If
only the endpoints land, the strongest reachable readings are **B**, **D** and
**E**, all of which are statements about a single rung.

## 5. Power analysis on the pilot SD, before the data land (§1.8)

Pilot: the five per-seed `settled − twin` differences already journalled on
`negation_scope` at `n_train = 8192` —

```
seed 0  -0.013821
seed 1  -0.015595
seed 2  +0.080153
seed 3  +0.018070
seed 4  -0.054011
mean +0.002959   sd 0.050146
```

(sign convention here is `NRMSE_settled − NRMSE_twin`, so **negative means settled
wins**. `scale/m3_synthetic_settled.py::contrast` reports the negation of this,
`delta = mean(twin − settled)`, where **positive means settled wins**. Both
conventions appear in this repo; every number below is labelled.)

Normal-approximation half-width `1.96 · sd / sqrt(N)`:

```
N =  5   0.043955       <- this run
N =  9   0.032762
N = 13   0.027260       <- the §1.8 e-process floor
N = 21   0.021448
N = 33   0.017110
```

**Resolution of this run: `|delta| ≥ 0.044` NRMSE.** A CI covering zero at `N = 5`
does **not** mean zero. It means "not distinguishable from zero at a resolution of
0.044".

## 6. The outcome table — fixed now, and it is what each result licenses

Read on `contrast()`'s convention (**positive delta = settled wins**), 95 % paired
percentile bootstrap, `B = 10000`, `seed = 0`, over the 5 seeds.

| # | what the data show | what it licenses |
|---|---|---|
| **A** | CI excludes zero and is **positive** at `t* = 32`, and covers zero at `t* = 1`, and the point estimate is **monotone non-decreasing** across `t* = 1, 2, 8, 32` | **DOSE-RESPONSE.** The one signature §1.7d calls believable. S1's `+12` route opens; the claim is *settling buys equilibrium capability that the identical mixture without the iteration does not*. |
| **B** | CI excludes zero and is positive at `t* = 32` but the ladder is **not** monotone | **PARTIAL.** A gap at one rung is not a dose-response. Reported as a single-rung effect with the non-monotonicity printed; does not open `+12`. |
| **C** | CI covers zero at **every** rung **and** `|delta| < 0.027` at every rung | **K-3E FIRES.** `LOOP_PROMPT.md` §4: *"E2/E3 solved equally by the twin at matched parameters ⇒ settling is capability-irrelevant on its own home terrain"*. Combined with §1.8's *"an effect that needs more than 13 seeds is reported as too small to matter at this scale and is not chased"*, the effect is below the chase floor. **Settling retires; the twin ships (§7).** |
| **D** | CI covers zero at every rung but `\|delta\| ≥ 0.027` at `t* = 8` or `t* = 32` | **UNDERPOWERED, NOT A KILL.** The point estimate is above the 13-seed floor, so §1.8 does not permit "too small to matter". Route: seeds 5→13 at that rung **only**, priced in the report. No kill is claimed. |
| **E** | CI excludes zero and is **positive at `t* = 1`** | **G4 SHAPE — capacity leakage**, not settling. §1.7d. The claim degrades to routing-only whatever `t* = 32` says. |
| **F** | CI excludes zero and is **negative** (settled loses) at `t* = 8` or `t* = 32`, **and** positive at `t* = 1` | **K-2E FIRES** as literally written in §4: *"Ladder E fails in both directions — settled loses on deep-`t*` E-tasks AND wins on `t* ≤ 1` tasks"*. Theory death; §2's five-minute question is asked. |
| **G** | any cell's `settled` or `twin` seed-mean NRMSE `≥ 1.0` at a rung | that rung is **credited nothing in either direction** — nothing beat predict-the-mean, so a contrast between two cells that both failed is a contrast between two failures. Printed, never read as a verdict. |
| **H** | CI excludes zero and is **negative** at some rung, and is never positive at any rung | **SETTLING IS A STRICT COST.** Not a tie and not a two-directional failure: the iteration measurably *hurts* at matched parameters. K-3E's conclusion holds a fortiori — the twin ships — and the retirement sentence is stronger than "bought nothing". |

**Row H was added at 13:35, after rows A–G and before any number from this
run existed, because A–G did not cover it.** Rows A, B, E and F all condition on
settled *winning* somewhere; C and D condition on every CI covering zero. A
ladder where settled only ever loses fell through all of them. A pre-registration
with a hole in it is not a pre-registration, and the hole is patched before the
data rather than after.

**On the brief's wording.** The dispatch called outcome C "K-2E". It is not.
K-2E as written in `LOOP_PROMPT.md:400` requires failure in *both* directions —
settled **losing** deep and **winning** shallow — which is row **F**. "Covers zero
everywhere" is *equal performance*, which is K-3E (`LOOP_PROMPT.md:403`). Both are
pre-registered above with their own rows so the distinction cannot be blurred after
the fact. The consequence for the product is the same either way: the twin ships.

## 7. The fallback product, written before any number is looked at

If row **C** or row **F** fires, this is what ships, and it is already earned on
`negation_scope` at `n_train = 8192`, 5 seeds, `results/m3_quintuple_v2.jsonl`:

> **Pivot-routed mixture attention (the `twin`).** `alpha` is the normalised gate
> over `k = 8` content-selected pivots; the pivot context is paid, the fixed-point
> loop is not.
>
> * `twin − softmax` = **+0.111396** NRMSE, CI **[+0.100873, +0.121920]**, 5/5
>   seeds favour the twin, `sd = 0.016547`.
> * The gain is the **mixture**, not a lookup: `argmax − softmax` = **−0.118456**,
>   CI **[−0.134115, −0.102786]**, 0/5 seeds. Collapsing the mixture to its
>   highest-weight pivot is worse than not routing at all.
> * The iteration is what is being retired, not the routing: `settled − twin` =
>   **−0.002959**, CI **[−0.042903, +0.031557]**, `sd 0.064106` against the twin's
>   `0.016547` — four times looser for no mean gain.

The retirement sentence is therefore *"the fixed point bought nothing; the mixture
bought +0.111396"*, and the deliverable is the twin module plus this table, not a
withdrawal.

## 7b. DISCLOSURE — the pilot rows that were visible before this table was closed

`LOOP_PROMPT.md` §1.8 permits the fixed-sample `N` to be set by power analysis
*on the pilot*, so looking at a pilot is legal. Hiding *which* pilot was looked
at is not. At 13:26–13:29, before this run had produced anything, the following
rows from another fellow's K-5E screen (`scale/etask_k5e.py`, seed 0 only, one
cell per row, no interval) were on disk and were read:

`results/etask_k5e.txt`, `n_train = 1024`, `n_eval = 2048`, seed 0:

```
       e3_t1    1   softmax   4769  1.002204  0.890073    BEATS BAR    83.2
       e3_t1    1      twin   4769  1.002255  0.966837    BEATS BAR   122.1
       e3_t1    1   settled   4769  1.002293  1.061138 AT/ABOVE BAR   188.7
       e3_t2    2   softmax   4769  1.003282  1.101813 AT/ABOVE BAR    79.2
       e3_t2    2      twin   4769  1.003317  1.077695 AT/ABOVE BAR   117.1
       e3_t2    2   settled   4769  1.003326  1.100449 AT/ABOVE BAR   190.3
       e3_t8    8   softmax   4769  1.007414  1.337387 AT/ABOVE BAR    84.3
       e3_t8    8      twin   4769  1.007418  1.273078 AT/ABOVE BAR   136.5
```

`results/etask_k5e_e1.txt`, `n_train = 512`, `n_eval = 2048`, seed 0:

```
   e1_anchor   63   settled   4769  1.000638  1.464949 AT/ABOVE BAR   115.8
   e1_anchor   63      twin   4769  1.000644  1.358240 AT/ABOVE BAR    67.2
   e1_anchor   63   softmax   4769  1.000655  1.644332 AT/ABOVE BAR    40.9
      e3_t32   32   settled   4769  1.000908  1.296438 AT/ABOVE BAR    95.5
      e3_t32   32      twin   4769  1.000876  1.302336 AT/ABOVE BAR    67.3
      e3_t32   32   softmax   4769  1.000847  1.423038 AT/ABOVE BAR    41.3
```

**What that changes, stated now so it cannot be claimed as a discovery later:**

1. **Row G is the expected outcome at `t* ≥ 2`.** At `n_train ≤ 1024` every cell
   at every rung above `t* = 1` sits *above* predict-the-mean. This run doubles
   `n_train` to 2048, which is unlikely to move 1.27 below 1.00. If row G fires
   at those rungs, the ladder is **unreadable at this budget** — and that is a
   statement about the ARMS' capacity, not about settling. It is not a kill for
   settling and this document does not permit it to be reported as one.
2. **Row H is live.** At `t* = 1`, `n_train = 1024`, seed 0, `settled` read
   `1.061138` against `twin`'s `0.966837` — settled on the wrong side of the
   bar while the twin cleared it. That single unreplicated point is what forced
   row H into the table. It is one seed at a different `n_train` with no
   interval and it decides nothing; it is disclosed because it was seen.
3. **The `t* = 8` two-hop floor is `0.815162`, not zero.** The truncation ladder
   printed by `scale/m3_capability.py --task e3_t8` at `n_train = 2048` reads
   `k=0 0.944013, k=1 0.880512, k=2 0.815162, k=4 0.674331, k=8 0.000000`
   (`results/m3_capability.txt:1212`). Every arm here has a hop budget of 2, so
   `0.815162` is the best NRMSE any of them could reach at `t* = 8` even with
   perfect training. It is below 1.0, so the rung is not structurally
   uncreditable — the arms are simply nowhere near their own ceiling at these
   budgets. `e3_t1`'s ladder reads `k=1 0.000000`
   (`results/m3_capability.txt:1280`), so `t* = 1` is exactly representable by
   a one-hop arm, which is why it is the rung that clears the bar.

**No change is made to the geometry, the seeds, the rungs or the outcome table
on the strength of these rows**, except the addition of row H, which adds a
losing branch rather than a winning one. The run proceeds as specified in §3.

## 8. Limits, collected once

* `N = 5`. Resolution 0.044 NRMSE. Section 5.
* The pilot SD is from `negation_scope` at `n_train = 8192`; this run is `e3_*` at
  `n_train = 2048`. The SD may differ and the resolution figure moves with it. The
  realised per-rung SD is printed beside every contrast.
* Every `e3` row inherits E1's rig with respect to `softmax`. Only `settled − twin`
  is read. Section 1.
* Wall-clock figures are PROVISIONAL: four other measurement processes were on this
  box during the run. No seconds figure enters a verdict sentence.
* `t* = 32` at `s = 64` is outside every arm's hop budget by construction. A null
  there is compatible with "settling does not help" **and** with "no arm in this
  family can reach 32 hops". This run cannot separate those two, and does not
  claim to.
